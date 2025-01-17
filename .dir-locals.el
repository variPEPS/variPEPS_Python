((python-mode
  (eval . (setq flycheck-python-flake8-executable
                (expand-file-name ".venv/bin/python" (locate-dominating-file
                                                      default-directory
                                                      ".dir-locals.el"))))
  (eval . (setq flycheck-python-pylint-executable
                (expand-file-name ".venv/bin/python" (locate-dominating-file
                                                      default-directory
                                                      ".dir-locals.el"))))
  (eval . (setq flycheck-python-mypy-executable
                (expand-file-name ".venv/bin/mypy" (locate-dominating-file
                                                    default-directory
                                                    ".dir-locals.el"))))
  (eval . (setq flycheck-python-pycompile-executable
                (expand-file-name ".venv/bin/python" (locate-dominating-file
                                                      default-directory
                                                      ".dir-locals.el")))))
 (rst-mode
  (eval . (setq flycheck-rst-sphinx-executable
                (expand-file-name ".venv/bin/sphinx-build" (locate-dominating-file
                                                            default-directory
                                                            ".dir-locals.el"))))))
