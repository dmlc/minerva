called=$_
if [[ $called != $0 ]]; then
  if [[ -z ${OWL_ENABLED} ]]; then
    SCRIPTPATH=$(dirname $(readlink -f "${BASH_SOURCE[@]}"))
    export OWL_ENABLED=true
    export PYTHONPATH=${SCRIPTPATH}/release/owl:${SCRIPTPATH}/owl:$PYTHONPATH
    export PS1="(Owl Ready) ${PS1}"
  else
    echo "Owl already enabled"
  fi
else
  echo "Script $0 should be sourced, not run. Use:" 1>&2
  echo "  . $0" 1>&2
fi
