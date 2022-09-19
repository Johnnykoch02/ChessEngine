import os
import sys
PROJECT_PATH = os.getcwd()
from src.Managers.AppManager import AppManager


def main ():
    print(PROJECT_PATH)
    AppManager(PROJECT_PATH)


if __name__ == '__main__':
    print('Hello World')
    main()


    