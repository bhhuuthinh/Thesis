import os

def GetDirectories(path = '.'):
    d_list = list()
    try:
        for f in os.listdir(path):
            if os.path.isdir(os.path.join(path, f)):
                    d_list.append(path + f + '/')
    except:
        return "\nError, once check the path"
    return d_list

def GetFiles(path = '.'):
    f_list = []
    try:
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)):
                f_list.append(path + f)
    except:
        return "\nError, once check the path"
    return f_list
