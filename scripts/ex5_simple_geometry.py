#
# Demo input file for parameterised geometries with Gmsh
# Author: Rory Spencer
# Date:  Nov 2024
# 
# Simple Geometry to update
# Elastic only model. 
# Target shape will be rectangular with height = 1, width = 0.8
# gaugeWidth parameter moves the width of the rectangular
# Model will have vonmises stress max of ~6E7 when neckWidth = 0.8

import gmsh
import argparse

parser = argparse.ArgumentParser("Config Parse")
parser.add_argument("-exportpath", help="Export path for this model.", type=str)
parser.add_argument("-neckWidth", help="Parameter 0.", type=float)
args = parser.parse_args()

#Geometry variables
gaugeHeight = 1
gaugeWidth = 1

neckWidth = args.neckWidth
lc = 0.2

gmsh.initialize()

gmsh.model.add("mod")


# Create some points defining the boundary
# Will have vertical symmetry
p1 = gmsh.model.geo.addPoint(0,0,0,lc) 
p2 = gmsh.model.geo.addPoint(neckWidth,0,0,lc)
p3 = gmsh.model.geo.addPoint(gaugeWidth,gaugeHeight,0,lc) 
p4 = gmsh.model.geo.addPoint(0,gaugeHeight,0,lc)

l1 = gmsh.model.geo.addLine(p1,p2)
l2 = gmsh.model.geo.addLine(p2,p3)
l3 = gmsh.model.geo.addLine(p3,p4)
l4 = gmsh.model.geo.addLine(p4,p1)

cl1 = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4])
s1 = gmsh.model.geo.addPlaneSurface([cl1])
print(s1)
gmsh.model.geo.mesh.setRecombine(2,s1)

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(2)

gmsh.model.geo.addPhysicalGroup(2, [s1], name='Specimen')
gmsh.model.geo.addPhysicalGroup(1, [l3], name='Top-BC')
gmsh.model.geo.addPhysicalGroup(1, [l4], name='X-Symm')
gmsh.model.geo.addPhysicalGroup(1, [l1], name='Y-Symm')

gmsh.model.geo.synchronize()

exportpath = args.exportpath + '.msh'
gmsh.write(exportpath)
gmsh.finalize()
