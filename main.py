from src.Utils.imports import os
import sys
PROJECT_PATH = os.getcwd()

def extract_command(command:str):
    cmd = command.split(' ')[0]
    arguments = command.replace(cmd,'').split(' -')
    args = {}
    for i in arguments:
        try:
            tmp = i.replace(' ', '#',1).split('#')
            args[tmp[0]] = tmp[1].strip()
        except:
            pass
    return cmd, args

commands = {'exit': {'fnc': sys.exit , 'args':['load_dir', 'num_steps']},
            'train': {'fnc': None , 'args':['load_dir', 'num_steps']},
            'demo': {'fnc': None, 'args':['model_loc']},
            'play': {'fnc': None, 'args':None},
            'data': {'fnc': None, 'args':None},
            'pretrain': {'fnc': None, 'args':None},
            'mc_simulate': {'fnc': None, 'args':None},
            }

def main():
    global commands
    from src.Utils.imports import AppManager
    print(PROJECT_PATH)
    iAppManager = AppManager(PROJECT_PATH)
    command = input("Input Command w/ Arguments: ")
    cmd, args = extract_command(command)

    if cmd in commands:
        if cmd == 'train':
            current_command = commands[cmd]
            if 'num_steps' in args and 'load_dir' in args:
                iAppManager.Train(int(args['num_steps']), args['load_dir'])
            elif len(args) == 0:
                iAppManager.Train()
            else:
                iAppManager.Train(num_steps=int(args['num_steps'])) if 'num_steps' in args else iAppManager.Train(load_dir= args['load_dir'])
        elif cmd == 'demo':
            current_command = commands[cmd]
        
        elif cmd == 'play':
            iAppManager.Run()
        
        elif cmd == 'data':
            iAppManager.set_DataCollection()
            iAppManager.Run()
        
        elif cmd == 'pretrain':
            iAppManager.set_PreTrain()
        
        elif cmd == 'mc_simulate':
            iAppManager.MCSimulate()
        



if __name__ == '__main__':
    main()


    