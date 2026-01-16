from IPython.display import Image, display
from random import randint

def logo():
    banner = """
      ____  _   _   _____          _                 _\/_
     / ___|| \ | | |  ___|_ _  ___| |_ ___  _ __ _   _/\ 
     \___ \|  \| | | |_ / _` |/ __| __/ _ \| '__| | | |
      ___) | |\  | |  _| (_| | (__| || (_) | |  | |_| |
     |____/|_| \_| |_|  \__,_|\___|\__\___/|_|   \__, |
                                                 |___/ 
                                            
                Baking particles since 1987!
                         v.beta.1
    """
#    """
#      ____  _   _   _____          _                  
#     / ___|| \ | | |  ___|_ _  ___| |_ ___  _ __ _   _\/_
#     \___ \|  \| | | |_ / _` |/ __| __/ _ \| '__| | | |\
#      ___) | |\  | |  _| (_| | (__| || (_) | |  | |_| |
#     |____/|_| \_| |_|  \__,_|\___|\__\___/|_|   \__, |
#                                                 |___/ 
#                                            
#Welcome to the Supernova Factory! Baking particles since 1987!
#    """
    print(banner)


def logoIMG():
    display( Image(filename='../Files/Logos/SNfactoryLogo_' + str(randint(0, 10)) + '.jpeg')  )
