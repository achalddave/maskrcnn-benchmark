# Usage: . ./env.sh (activate|deactivate)
# Hacky script to add/remove this directory to PYTHONPATH.

function usage() {
    echo "Usage:"
    echo ". ./env.sh (activate|deactivate)"
}

if [[ $_ == $0 ]] ; then
    echo "This script needs to be sourced."
    usage
    exit 1
fi

if [[ "$#" != 1 ]] ; then
    usage
    return
elif [[ "$1" != "activate" ]] && [[ "$1" != "deactivate" ]] ; then
    usage
    return
fi

_PROJECT_DIR="$( cd "$( dirname "$0" )" >/dev/null && pwd )"
if [[ "$1" == "activate" ]] ; then
    if [[ -n "${PYTHONPATH}" ]] ; then
        export PYTHONPATH="${_PROJECT_DIR}:${PYTHONPATH}"
    else
        export PYTHONPATH="${_PROJECT_DIR}"
    fi
    unset _PROJECT_DIR
elif [[ "$1" == "deactivate" ]] ; then
    export PYTHONPATH=${PYTHONPATH//:${_PROJECT_DIR}/}
    export PYTHONPATH=${PYTHONPATH//${_PROJECT_DIR}/}
fi
unset _PROJECT_DIR
