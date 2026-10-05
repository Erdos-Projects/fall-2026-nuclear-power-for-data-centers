# Contributing

## Getting started

Clone the shared repository and follow the installation instructions
in README.md.

Before starting a task, create or claim a GitHub Issue so that we
avoid duplicating work.

## Working on changes

Update your local main branch and create a task branch:

    git switch main
    git pull --ff-only origin main
    git switch -c your-name/task-name

Use descriptive branch names, such as:
- member-name/nrc-download
- member-name/outage-definitions
- member-name/duration-model

Keep changes focused on one task.

## Submitting changes

Commit and push your changes:

    git add path/to/changed-file
    git commit -m "Describe the change"
    git push -u origin your-name/task-name

Open a pull request on GitHub. Explain:
- What changed and why.
- How you checked the changes.
- Whether the changes affect shared data or definitions.

Ask another team member to review the pull request before merging.
Use pull requests for changes to main.

## Data conventions

- Preserve raw source files unchanged in data/raw/.
- Record source URLs, retrieval times and file hashes.
- Generate processed data using committed scripts.
- Document column names, units and missing values in data/README.md.
- Do not treat missing observations as shutdowns.
- Discuss changes to outage definitions with the team.

## Notebooks

Use descriptive notebook names and avoid having multiple people
edit the same notebook at the same time. Clear unnecessary outputs
before committing. Put reusable code in src/.

## Credentials

Never commit passwords, API keys or other credentials.