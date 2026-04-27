# AI-Assisted Development

This project uses an AI-first development workflow. Most features, bug fixes, and improvements are implemented through AI coding agents (Claude Code, GitHub Copilot, OpenAI Codex) working from structured issue descriptions.

## How It Works

1. **Issues are the starting point** — Every task begins as a GitHub issue with a structured format: an overview, a human-readable plan, a detailed implementation plan (in a collapsible block), and optionally the original prompt that generated the issue.

2. **AI agents pick up issues** — Issues can be assigned to AI coding agents (e.g. GitHub Copilot) which read the issue description, `AGENTS.md`, and `CLAUDE.md` for context, then implement the changes autonomously.

3. **Human review** — All AI-generated pull requests are reviewed by maintainers before merging.

## Maintainer Workflow

Maintainer-driven dev work starts as a prompt file in
**[PyAutoPrompt](https://github.com/PyAutoLabs/PyAutoPrompt)** — the public
workflow repo that hosts the PyAuto task registry and the prompt-coupled
Claude Code skills. The pipeline:

1. Write the task as `PyAutoPrompt/<category>/<name>.md` (free-form markdown
   describing what to do, with `@RepoName/path/to/file.py` references).
2. `/start_dev <category>/<name>.md` — reads the prompt, audits the code,
   drafts the GitHub issue you see in this repo, and files it.
3. `/start_library` or `/start_workspace` — opens a feature worktree under
   `~/Code/PyAutoLabs-wt/<task-name>/`.
4. `/ship_library` / `/ship_workspace` — runs tests, opens the PR, and
   tracks state in `PyAutoPrompt/active.md`.

External contributors don't need PyAutoPrompt access — open an issue using
the templates in this repo and the same machinery handles it on our end.

## Creating an Issue

When opening an issue, please use the provided issue templates. The **Feature / Task Request** template follows our standard format:

- **Overview** — What and why, in 2-4 sentences
- **Plan** — High-level bullet points (human-readable)
- **Detailed implementation plan** — File paths, steps, key files (in a collapsible block)
- **Original Prompt** — If you used an AI to help draft the issue, include the original prompt

If your feature involves a specific calculation, algorithm, or small piece of functionality — **include example code**. Even a rough script, a working prototype, or a snippet showing the existing behaviour you want to change makes a huge difference. Code examples give AI agents and human contributors concrete context to work from, and dramatically reduce misunderstandings about what you're asking for.

This structure ensures that both human contributors and AI agents can understand and act on the issue effectively.

## Contributing Without AI

Traditional contributions are equally welcome! If you prefer to work without AI tools, simply follow the development setup and pull request guidelines below. The issue templates are helpful for any contributor, AI or human.

---

## Community Guidelines

We strive to maintain a welcoming, respectful, and inclusive community. All contributors—whether opening issues, submitting pull requests, reviewing code, or participating in discussions—are expected to follow these guidelines.

- Be respectful and considerate of others.
- Assume good intent and be patient, especially with newcomers.
- Keep feedback constructive and focused on the work, not the person.
- Communicate clearly and professionally.
- Respect maintainers’ time and decisions.

Harassment, discrimination, or abusive behavior of any kind will not be tolerated.

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.

### Reporting concerns
If you experience or witness behavior that violates these guidelines or the Code of Conduct, please contact the project maintainers privately.


# Contributing

Contributions are welcome and greatly appreciated!

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/Jammy2211/PyAutoLens/issues

If you are playing with the PyAutoLens library and find a bug, please
reporting it including:

* Your operating system name and version.
* Any details about your Python environment.
* Detailed steps to reproduce the bug.

### Propose New Features

The best way to send feedback is to open an issue at
https://github.com/Jammy2211/PyAutoLens
with tag *enhancement*.

If you are proposing a nnew feature:

* Explain in detail how it should work.
* Keep the scope as narrow as possible, to make it easier to implement.

### Implement Features
Look through the Git issues for operator or feature requests.
Anything tagged with *enhancement* is open to whoever wants to
implement it.

### Add Examples or improve Documentation
Writing new features is not the only way to get involved and
contribute. Create examples with existing features as well 
as improving the documentation of existing operators is as important
as making new non-linear searches and very much encouraged.


## Getting Started to contribute

Ready to contribute?

1. Follow the installation instructions for installing **PyAutoLens** (and parent projects) from source root 
on our [readthedocs](https://pyautolens.readthedocs.io/en/latest/installation/source.html).

2. Create a feature branch for local development (for **PyAutoLens** and every parent project where changes are implemented):
    ```
    git checkout -b feature/name-of-your-branch
    ```
    Now you can make your changes locally.

3. When you're done making changes, check that old and new tests pass successfully:
    ```
    cd PyAutoLens/test_autolens
    python3 -m pytest
    ```

4. Commit your changes and push your branch to GitHub::
    ```
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin feature/name-of-your-branch
    ```
    Remember to add ``-u`` when pushing the branch for the first time.

5. Submit a pull request through the GitHub website.


### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include new tests for all the core routines that have been developed.
2. If the pull request adds functionality, the docs should be updated accordingly.
