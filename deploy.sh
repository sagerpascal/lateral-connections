for HOST in 'ubuntu@apu2038:~/' 'ubuntu@apu3235:~/' 'sage@dgx.cloudlab.zhaw.ch:~/'
do
  for FOLDER in 'src/' 'configs/'
  do
    rsync -av -e ssh -I --exclude='.wandb/*' $FOLDER $HOST\lateral_connections/$FOLDER
  done
done