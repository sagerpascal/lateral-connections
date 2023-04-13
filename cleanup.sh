# Create folders if they do not exist
for FOLDER in 'plots/' 'wandb/' 'data/' 'checkpoints/'
do
  mkdir -p $FOLDER
done

# Delete all files in the plots and wandb folders
for FOLDER in 'plots/' 'wandb/'
do
  rm -r $FOLDER
  mkdir $FOLDER
done