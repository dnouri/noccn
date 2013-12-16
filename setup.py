import os

from setuptools import setup, find_packages

version = '0.1-dev'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
    CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()
except IOError:
    README = CHANGES = ''

install_requires = [
    'joblib',
    ]

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    ]

docs_require = [
    'Sphinx',
    ]

setup(name='noccn',
      version=version,
      description="Utilities for Alex Krizhevsky's cuda-convnet",
      long_description='\n\n'.join([README, CHANGES]),
      classifiers=[
          'Development Status :: 3 - Alpha',
        ],
      keywords='',
      author='Daniel Nouri',
      author_email='daniel.nouri@gmail.com',
      url='https://github.com/dnouri/noccn',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      entry_points="""\
      [console_scripts]
      ccn-predict = noccn.predict:console
      ccn-train = noccn.train:console
      ccn-show = noccn.show:console
      ccn-make-batches = noccn.dataset:console
      """,
      extras_require={
          'testing': tests_require,
          'docs': docs_require,
          },
      )
