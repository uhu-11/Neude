import inspect

from pythonfuzz import tracer
from pythonfuzz.Coverages.Coverage import Coverage
from pythonfuzz.utils import htmlpaser



class LineCoverage():

    def get_line_coverage(self, folder_path):
        return htmlpaser.coveraged_line(folder_path=folder_path)


   
    def get_line_coverage_num(self):
        return tracer.get_coverage()


    