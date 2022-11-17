import gym
import minihack
import imageio
import os
import shutil
import numpy as np
from PIL import Image
import torchvision
import cv2

def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image

def get_dataset(env_list, num_screenshots=10000, generic_dir_name = 'numerical_expts'):
    
    np.set_printoptions(threshold = np.inf)

    dataset_path = 'minihack_datasets/{0}/'.format(generic_dir_name) 
    output_path = 'outputs'

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    pixels_path = dataset_path + 'pixels/'
    glyphs_path = dataset_path + 'glyphs/'
    messages_path = dataset_path + 'messages/'
    screen_descriptions_paths = dataset_path + 'screen_descriptions/'

    os.makedirs(pixels_path)
    os.makedirs(glyphs_path)
    os.makedirs(messages_path)
    os.makedirs(screen_descriptions_paths)
    
    counter = 0
    for env_name in env_list:
        env = gym.make(env_name, observation_keys=("glyphs", "pixel", "message", "screen_descriptions"))
        env.reset()
        for _ in range(num_screenshots):
            npy_path = '{}.npy'.format(counter)
            glyphs_file = open(glyphs_path + npy_path, "wb")
            messages_file = open(messages_path + npy_path, "wb")
            screen_descriptions_file = open(screen_descriptions_paths + npy_path, "wb")

            action = env.action_space.sample()
            obs, rewards, done, info = env.step(action)
            if done:
                env.reset()
            img_dataset_path = pixels_path + '/{0}.jpg'.format(counter)

            img = obs['pixel']
            #img = autocrop(img)

            imageio.imwrite(img_dataset_path, img)
            np.save(glyphs_file, obs['glyphs'])
            np.save(messages_file, obs['message'])
            np.save(screen_descriptions_file, obs['screen_descriptions'])
            counter += 1

    glyphs_file.close()
    messages_file.close()
    screen_descriptions_file.close()


def main():
    env_list = ['MiniHack-Room-Ultimate-15x15-v0',
    'MiniHack-River-Monster-v0',
    'MiniHack-River-v0',
    'MiniHack-Corridor-R5-v0',
    'MiniHack-KeyRoom-S15-v0',
    'MiniHack-KeyRoom-Dark-S15-v0',
    'MiniHack-MazeWalk-Mapped-45x19-v0',
    'MiniHack-River-MonsterLava-v0',
    'MiniHack-HideNSeek-Lava-v0',
    'MiniHack-HideNSeek-Big-v0',
    'MiniHack-CorridorBattle-v0',
    'MiniHack-CorridorBattle-Dark-v0',
    'MiniHack-Memento-F4-v0',
    'MiniHack-ExploreMaze-Hard-Mapped-v0']

    '''
    get_dataset('MiniHack-River-Monster-v0')
    get_dataset('MiniHack-River-v0')
    '''

    get_dataset(env_list)
    

if __name__ == '__main__':
    main()
