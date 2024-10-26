python3 process.py --input_file ../dataset/biored/Train.PubTator \
                   --output_file ../dataset/biored_cd/train \
                   --data biored_cd
python3 process.py --input_file ../dataset/biored/Dev.PubTator \
                   --output_file ../dataset/biored_cd/dev \
                   --data biored_cd
python3 process.py --input_file ../dataset/biored/Test.PubTator \
                   --output_file ../dataset/biored_cd/test \
                   --data biored_cd

cat ../dataset/biored_cd/train.data ../dataset/biored_cd/dev.data > ../dataset/biored_cd/train+dev.data

python3 process_cd.py --input_file ../dataset/biored/Test.PubTator \
                      --output_file ../dataset/biored_cd/Test.pubtator