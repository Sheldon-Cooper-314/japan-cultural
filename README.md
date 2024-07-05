# t-SNE for Feature Visualization

# Executing the T-SNE visualization

```bash

python3 tsne.py

```

Additional options:

```bash
python3 tsne.py -h

usage: tsne.py [-h] [--path PATH] [--batch BATCH] [--num_images NUM_IMAGES]

optional arguments:
  -h, --help            show this help message and exit
  --path PATH
  --batch BATCH
  --num_images NUM_IMAGES

```

You can change the data directory with `--path` argument.

Tweak the `--num_images` to speed-up the process - by default it is 500, you can make it smaller.

Tweak the `--batch` to better utilize your PC's resources. The script uses GPU automatically if it available. You may
want to increase the batch size to utilize the GPU better or decrease it if the default batch size does not fit your
GPU.
