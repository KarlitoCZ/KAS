import pyglet
import shared
from dataclasses import dataclass

@dataclass
class Vector:
    x: int
    y: int

    def add(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
@dataclass   
class RGB:
    r: int
    g: int
    b: int

    def add(self, other):
        return RGB(self.x + other.x, self.y + other.y)
    
def draw_line(pos1 : Vector, pos2 : Vector, color : RGB):
    shared.objects.append(pyglet.shapes.Line(pos1.x, pos1.y, pos2.x, pos2.y, color=(color.r, color.g, color.b), batch=shared.batch))
    print("Line added")
    print(shared.objects)
    #return pyglet.shapes.Line(pos1.x, pos1.y, pos2.x, pos2.y, color=(color.r, color.g, color.b), batch=shared.batch)

def draw_box(pos1 : Vector, pos2 : Vector, color : RGB):
    #draw_line(Vector(250,250), Vector(600,700), RGB(0,250,0))
    print("Box")
    #return pyglet.shapes.Line(pos1.x, pos1.y, pos2.x, pos2.y, color=(color.r, color.g, color.b), batch=shared.batch)