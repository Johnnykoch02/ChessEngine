import os
PROJECT_PATH = os.getcwd()


def main ():
    from src.Managers.AppManager import AppManager
    print(PROJECT_PATH)
    AppManager(PROJECT_PATH)


if __name__ == '__main__':
    print('Hello World')
    main()


    