import cocopp
import os
import webbrowser

cocopp.main('cma-fmin2-001.tgz scipy-optimize-fmin.tgz')
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")