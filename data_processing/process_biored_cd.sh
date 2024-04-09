python3 process.py --input_file ../data/biored/Train.PubTator \
                   --output_file ../data/biored/processed/train \
                   --data biored_cd
python3 process.py --input_file ../data/biored/Dev.PubTator \
                   --output_file ../data/biored/processed/dev \
                   --data biored_cd
python3 process.py --input_file ../data/biored/Test.PubTator \
                   --output_file ../data/biored/processed/test \
                   --data biored_cd