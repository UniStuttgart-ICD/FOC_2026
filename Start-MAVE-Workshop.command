#!/usr/bin/env bash
set -u

cd "$(dirname "$0")" || exit 1

install_ur_rtde=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-ur-rtde)
      install_ur_rtde=1
      ;;
    -h|--help)
      echo "Usage: ./Start-MAVE-Workshop.command [--with-ur-rtde]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./Start-MAVE-Workshop.command [--with-ur-rtde]"
      exit 1
      ;;
  esac
  shift
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found on PATH."
  echo "Install uv, then run this launcher again."
  exit 1
fi

sync_args=(sync)
if [ "$install_ur_rtde" -eq 1 ]; then
  sync_args+=(--extra robot)
fi

if [ ! -x "server/.venv/bin/python" ] || [ "$install_ur_rtde" -eq 1 ]; then
  if [ ! -x "server/.venv/bin/python" ]; then
    echo "Creating the server environment..."
  fi
  if [ "$install_ur_rtde" -eq 1 ]; then
    echo "Installing ur-rtde robot extra..."
  fi
  (
    cd server || exit 1
    uv "${sync_args[@]}"
  )
  sync_status=$?
  if [ "$sync_status" -ne 0 ]; then
    echo "uv sync failed."
    exit "$sync_status"
  fi
fi

(
  cd server || exit 1
  uv run python -m operator_dashboard
)
exit_code=$?

if [ "$exit_code" -ne 0 ]; then
  echo "Operator dashboard exited with code $exit_code."
fi

exit "$exit_code"
