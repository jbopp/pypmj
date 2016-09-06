# The project library

jcmpython supports the usage of a *project library*, i.e. a `projects` folder at one location in your file system from where your projects are loaded. The path to this directory can be set in the configuration file and instances of `JCMProject` can then be created using a path relative to the project library. The `JCMProject`-instance will copy the files into a working directory, so that no changes are made to the actual project files.
This can be useful to organize your projects in a logical way, but also to share projects with others or among different machines.

## The shipped library

The projects shipped with jcmpython include minimal examples which are also used by test routines and example notebooks, but should also contain more real world content. *You can help* by providing your JCMsuite project, so that others can learn from your work. Optimally, the project is included in a subfolder which describes/classifies the kind of the project.

## Descriptive content

Besides the jcm-files, each project folder recommended to include a `README.md` which gives a description of the project, its parameters and perhaps processing functions using the [mark down language](http://daringfireball.net/projects/markdown/). You can use the `README_TEMPLATE.md` in this directory as a basis. The `JCMProject` class has its own `show_readme()`-method, which will display a parsed version inside the ipython/jupyter notebook or return the content as a string.


