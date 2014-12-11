SCRIPTPATH="$(cd $(dirname $0); pwd -P)"
export PYTHONPATH=${SCRIPTPATH}/release/owl:$PYTHONPATH
