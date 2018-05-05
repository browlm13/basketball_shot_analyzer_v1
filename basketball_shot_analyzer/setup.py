from setuptools import setup

setup(name='basketball_shot_analyzer',
      version='0.1',
      description='NULL',
      url='https://github.com/browlm13/basketball_shot_analyzer_v1.git',
      author='LJ Brown',
      author_email='browlm13@gmail.com',
      # license='',
      packages=['basketball_shot_analyzer'],
      install_requires=[
          'glob', 'numpy', 'cv2', 'matplotlib', 'scipy', 'piecewise'
      ],
      zip_safe=False)