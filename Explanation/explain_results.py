import os,sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

PROGRAM_PATH = '/net/home/aiqy/Project/KnowledgeEmbedding/ProductSearch/'
DATA_PATH = '/mnt/scratch/aiqy/ReviewEmbedding/working/Amazon/reviews_Cell_Phones_and_Accessories_5.json.gz.stem.nostop/min_count5/'
MODEL_DIR = '/mnt/scratch/aiqy/KnowledgeEmbedding/output/final_results/ProductSearch/CellPhone_bias_product/' 

command = 'python ' + PROGRAM_PATH + 'main.py --data_dir ' + DATA_PATH
command += ' --train_dir ' + MODEL_DIR

INPUT_TRAIN_DIR = DATA_PATH + 'query_split/'
command += ' --input_train_dir ' + INPUT_TRAIN_DIR

command += ' --learning_rate 0.5'
command += ' --net_struct fs'
command += ' --L2_lambda 0.00'
command += ' --steps_per_checkpoint 100'
#command += ' --seconds_per_checkpoint 10'
command += ' --subsampling_rate 0.0'
command += ' --max_train_epoch 10'
command += ' --batch_size 64'
command += ' --embed_size 200'
command += ' --similarity_func bias_product'

print(command + ' --decode true' + ' --test_mode explain')
os.system(command + ' --decode true' + ' --test_mode explain')