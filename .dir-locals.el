((nil . ((eval . (setq default-directory
                       (locate-dominating-file default-directory
                                               ".dir-locals.el")))))
 (python-mode
  (flycheck-python-flake8-executable . "venv/bin/python")
  (flycheck-python-pylint-executable . "venv/bin/pylint")
  (flycheck-python-mypy-executable . "venv/bin/mypy")
  (flycheck-python-pycompile-executable . "venv/bin/python"))
 (rst-mode
  (flycheck-rst-sphinx-executable . "venv/bin/sphinx-build")))
