if [[ -z ${OWL_ENABLED} ]]; then
  SCRIPTPATH=$(dirname `readlink -f -- $0`)
  export OWL_ENABLED=true
  export PYTHONPATH=${SCRIPTPATH}/owl:$PYTHONPATH
  export PS1="(Owl Ready) ${PS1}"
  SCRIPTPATH=
else
  echo "Owl already enabled"
fi
