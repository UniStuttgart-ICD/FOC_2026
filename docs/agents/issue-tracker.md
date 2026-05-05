# Issue tracker: GitHub Enterprise

Issues and PRDs for this repo live in GitHub Enterprise issues:

`https://github.tik.uni-stuttgart.de/ac147490/Robot_buddy`

Use the `gh` CLI for issue operations. The repo remote is configured as `origin`, so `gh` can infer the repository when run inside this clone. If inference fails, pass `-R github.tik.uni-stuttgart.de/ac147490/Robot_buddy`.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..." --label needs-triage`
- **Read an issue**: `gh issue view <number> --comments --json number,title,body,labels,comments,state`
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments`
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

## When a skill says "publish to the issue tracker"

Create a GitHub Enterprise issue in `ac147490/Robot_buddy` and apply `needs-triage` unless the skill specifies another triage label.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments --json number,title,body,labels,comments,state`.
