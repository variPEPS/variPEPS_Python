import datetime
import os
import subprocess


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
        ):
            if (entry := job_data.get(field)) is not None:
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

        if (entry := job_data.get("TRES")) is not None:
            entry = entry.split(",")
            if len(entry) > 0:
                job_data["TRES"] = dict(e.split("=", 1) for e in entry)

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
