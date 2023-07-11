import logging

def setup_log(path : str) -> None:
    logging.basicConfig(filename=path,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG,
                            force=True)