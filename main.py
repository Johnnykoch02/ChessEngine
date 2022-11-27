from src.Utils.imports import os
PROJECT_PATH = os.getcwd()


def main ():
    from src.Utils.imports import AppManager
    print(PROJECT_PATH)
    AppManager(PROJECT_PATH)


if __name__ == '__main__':
    main()


    