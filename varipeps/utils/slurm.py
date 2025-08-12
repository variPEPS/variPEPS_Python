import datetime
import math
import os
import pathlib
import sys
import subprocess
import textwrap


class SlurmUtils:
    @staticmethod
    def parse_special_fields(job_data):
        job_data = job_data.copy()

        for field in (
            "SubmitTime",
            "EligibleTime",
            "AccrueTime",
            "StartTime",
            "EndTime",
            "LastSchedEval",
            "PreemptEligibleTime",
        ):
            if (entry := job_data.get(field)) is not None and entry != "Unknown":
                job_data[field] = datetime.datetime.fromisoformat(entry)

        for field in ("RunTime", "TimeLimit", "DelayBoot"):
            if (entry := job_data.get(field)) is not None:
                entry = entry.split("-")
                if len(entry) == 2:
                    days, time = entry
                else:
                    (time,) = entry
                    days = 0

                hours, minutes, seconds = time.split(":")

                job_data[field] = datetime.timedelta(
                    days=int(days),
                    hours=int(hours),
                    minutes=int(minutes),
                    seconds=int(seconds),
                )

        for field in ("TRES", "ReqTRES", "AllocTRES"):
            if (entry := job_data.get(field)) is not None:
                entry = entry.split(",")
                if len(entry) > 0:
                    job_data[field] = dict(e.split("=", 1) for e in entry)

        for field in job_data:
            try:
                entry = int(job_data[field])
                job_data[field] = entry
            except (ValueError, TypeError):
                pass

            try:
                entry = job_data[field]
                if not isinstance(entry, int):
                    entry = float(entry)
                    job_data[field] = entry
            except (ValueError, TypeError):
                pass

        return job_data

    @classmethod
    def get_job_data(cls, job_id):
        job_id = int(job_id)

        p = subprocess.run(
            ["scontrol", "show", "job", f"{job_id:d}"], capture_output=True, text=True
        )

        if p.returncode != 0:
            return None

        job_data = p.stdout.split()

        slice_comb_list = []
        for i, e in enumerate(job_data):
            if "=" not in e:
                slice_comb_list[-1] = slice(slice_comb_list[-1].start, i + 1)
            else:
                slice_comb_list.append(slice(i, i + 1))

        job_data = ["".join(job_data[s]) for s in slice_comb_list]
        job_data = dict(e.split("=", 1) for e in job_data)

        job_data = cls.parse_special_fields(job_data)

        return job_data

    @classmethod
    def get_own_job_data(cls):
        if (job_id := os.environ.get("SLURM_JOB_ID")) is not None:
            return cls.get_job_data(job_id)
        return None

    @staticmethod
    def run_slurm_script(path, cwd=None):
        cwd = pathlib.Path(cwd).absolute()

        prog_env = {
            k: v
            for k, v in os.environ.items()
            if not (k.startswith("SBATCH_") or k.startswith("SLURM_"))
        }

        p = subprocess.run(
            ["sbatch", "--export=NONE", str(path)],
            capture_output=True,
            text=True,
            cwd=cwd,
            env=prog_env,
        )

        if p.returncode != 0:
            return None

        try:
            job_id = int(p.stdout.split()[-1])
        except ValueError:
            job_id = None

        return job_id

    @staticmethod
    def generate_restart_scripts(
        slurm_script_path,
        python_script_path,
        restart_state_file,
        slurm_data,
        executable=None,
    ):
        TEMPLATE_PYTHON = textwrap.dedent(
            """\
        #!/usr/bin/env python3
        import argparse
        import pathlib
        import varipeps

        parser = argparse.ArgumentParser()
        parser.add_argument('filename', type=pathlib.Path)
        args = parser.parse_args()

        varipeps.optimization.restart_from_state_file(args.filename)
        """
        )

        TEMPLATE_SLURM = textwrap.dedent(
            """\
        #!/bin/bash

        #SBATCH --partition={partition}
        #SBATCH --qos={qos}
        #SBATCH --job-name={job_name}
        #SBATCH --ntasks={ntasks:d}
        #SBATCH --nodes={nodes:d}
        #SBATCH --cpus-per-task={ncpus:d}
        #SBATCH --mem={mem}
        #SBATCH --time={time_limit}
        {mail_type}
        {mail_user}

        "{executable}" "{python_script}" "{state_file}"
        """
        )

        python_script_path = pathlib.Path(python_script_path).resolve()

        restart_state_file = pathlib.Path(restart_state_file).resolve()

        if executable is None:
            executable = pathlib.Path(sys.executable).absolute()

        if (tres := slurm_data.get("ReqTRES")) is not None:
            mem = tres["mem"]
        else:
            tres = slurm_data["TRES"]
            mem = tres["mem"]

        time_limit = slurm_data["TimeLimit"]
        if time_limit.days > 0:
            time_limit_diff = time_limit - datetime.timedelta(days=time_limit.days)
        else:
            time_limit_diff = time_limit

        time_limit_hours = math.floor(time_limit_diff / datetime.timedelta(hours=1))
        time_limit_diff -= datetime.timedelta(hours=time_limit_hours)

        time_limit_minutes = math.floor(time_limit_diff / datetime.timedelta(minutes=1))
        time_limit_diff -= datetime.timedelta(minutes=time_limit_minutes)

        time_limit_seconds = math.floor(time_limit_diff.total_seconds())

        if time_limit.days > 0:
            time_limit_str = f"{time_limit.days}-{time_limit_hours:02d}:{time_limit_minutes:02d}:{time_limit_seconds:02d}"
        else:
            time_limit_str = f"{time_limit_hours:02d}:{time_limit_minutes:02d}:{time_limit_seconds:02d}"

        if (mail_type := slurm_data.get("MailType")) is not None:
            mail_type = f"#SBATCH --mail-type={mail_type}"
        else:
            mail_type = ""

        if (mail_user := slurm_data.get("MailUser")) is not None:
            mail_user = f"#SBATCH --mail-user={mail_user}"
        else:
            mail_user = ""

        slurm_file_content = TEMPLATE_SLURM.format(
            partition=slurm_data["Partition"],
            qos=slurm_data["QOS"],
            job_name=f"{slurm_data['JobName']}_restarted",
            ntasks=slurm_data["NumTasks"],
            nodes=slurm_data["NumNodes"],
            ncpus=slurm_data["CPUs/Task"],
            mem=mem,
            time_limit=time_limit_str,
            mail_type=mail_type,
            mail_user=mail_user,
            executable=str(executable),
            python_script=str(python_script_path),
            state_file=restart_state_file,
        )

        with python_script_path.open("w") as f:
            f.write(TEMPLATE_PYTHON)

        with pathlib.Path(slurm_script_path).open("w") as f:
            f.write(slurm_file_content)
