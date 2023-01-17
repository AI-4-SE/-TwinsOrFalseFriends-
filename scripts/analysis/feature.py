from typing import List


class Feature:
    name = ""
    binary = False
    mandatory = None
    strictly_mandatory = None
    alternative_parent = False
    alternatives = []
    parent = None
    children = []
    exclusions = []
    implications = []

    def __init__(self, name: str, parent, exclusions: List, implications: List, mandatory: bool = None) -> None:
        self.binary = mandatory is not None
        self.name = name
        self.mandatory = mandatory
        self.parent = parent
        self.implications = implications
        self.exclusions = exclusions

    def __str__(self):
        return f"({self.name}, {str(self.binary)}, {str(self.mandatory)})"
