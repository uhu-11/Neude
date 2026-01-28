import inspect

from neude import tracer
from neude.Coverages.Coverage import Coverage
from neude.utils import htmlpaser



class LineCoverage():

    def get_line_coverage(self, folder_path):
        return htmlpaser.coveraged_line(folder_path=folder_path)


   
    def get_line_coverage_num(self):
        return tracer.get_coverage()


    