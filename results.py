import cocopp
import os
import webbrowser

cocopp.main('cma.tgz external-cma.tgz ego.tgz parEgo.tgz')
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")