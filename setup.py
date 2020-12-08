from distutils.core import setup
setup(
  name = 'climatenet',         # How you named your package folder (MyLib)
  packages = ['climatenet', 'climatenet.utils'],   # Chose the same as "name"
  version = 'v0.2',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A Python library for deep learning-based Climate Science.',   # Give a short description about your library
  author = 'Andre Graubner',                   # Type in your name
  author_email = 'andregr@ethz.ch',      # Type in your E-Mail
  url = 'https://github.com/andregraubner/ClimateNet',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/andregraubner/ClimateNet/archive/v0.2-alpha.tar.gz',    # I explain this later on
  keywords = [],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'cartopy',
          'matplotlib',
          'xarray',
          'tqdm',
          'torch',
          'haversine',
          'psutil'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)