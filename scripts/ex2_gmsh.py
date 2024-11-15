import gmsh
import sys
import argparse

parser = argparse.ArgumentParser("Config Parse")
parser.add_argument("-exportpath", help="Export path for this model.", type=str)
parser.add_argument("-p0", help="Parameter 0.", type=float)
parser.add_argument("-p1", help="Parameter 1.", type=float)
args = parser.parse_args()

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()

# Next we add a new model named "t1" (if gmsh.model.add() is not called a new
# unnamed model will be created on the fly, if necessary):
gmsh.model.add("circ")

vp0 = args.p0
vp1 = args.p1
gaugeHeight = 15
file_number = 1
gaugeWidth = 7.5
gaugeThickness = 1

s0 = -gaugeWidth-vp0
l0 = gaugeHeight*2
r0 = (s0/2) + ((l0*l0)/(8*s0))
s1 = gaugeWidth-vp1
l1 = gaugeHeight*2
r1 = (s1/2) + ((l1*l1)/(8*s1))
lc=1

p1 = gmsh.model.geo.addPoint(-gaugeWidth,gaugeHeight,0,lc) 
p2 = gmsh.model.geo.addPoint(gaugeWidth,gaugeHeight,0,lc)

p5 = gmsh.model.geo.addPoint(vp0,0,0,lc)
p6 = gmsh.model.geo.addPoint(vp0+r0,0,0,lc)

c1 = gmsh.model.geo.addCircleArc(p1,p6,p5)

p7 = gmsh.model.geo.addPoint(vp1,0,0,lc)
p8 = gmsh.model.geo.addPoint(vp1+r1,0,0,lc)

c2 = gmsh.model.geo.addCircleArc(p2,p8,p7)

l1 = gmsh.model.geo.addLine(p1,p2)
l3 = gmsh.model.geo.addLine(p7,p5)

p9 = gmsh.model.geo.addPoint(-gaugeWidth,gaugeHeight+2,0,1)
p10 = gmsh.model.geo.addPoint(gaugeWidth,gaugeHeight+2,0,1)

l8 = gmsh.model.geo.addLine(p2,p10)
l9 = gmsh.model.geo.addLine(p10,p9)
l10 = gmsh.model.geo.addLine(p9,p1)

cl1 = gmsh.model.geo.addCurveLoop([l1,c2,l3,-c1])
s1 = gmsh.model.geo.addPlaneSurface([cl1])
cl2 = gmsh.model.geo.addCurveLoop([l1,l8,l9,l10])
s2 = gmsh.model.geo.addPlaneSurface([cl2])

gmsh.model.geo.mesh.set_transfinite_curve(l1,10)
gmsh.model.geo.mesh.set_transfinite_curve(l3,10)
gmsh.model.geo.mesh.set_transfinite_curve(l9,10)

gmsh.model.geo.mesh.set_transfinite_curve(c1,10)
gmsh.model.geo.mesh.set_transfinite_curve(c2,10)
gmsh.model.geo.mesh.set_transfinite_curve(l8,3)
gmsh.model.geo.mesh.set_transfinite_curve(l10,3)

gmsh.model.geo.mesh.set_transfinite_surface(s1)
gmsh.model.geo.mesh.set_transfinite_surface(s2)

gmsh.model.geo.mesh.setRecombine(2,s1)
gmsh.model.geo.mesh.setRecombine(2,s2)

gmsh.model.geo.synchronize()

#gmsh.model.mesh.generate(2)

ov = gmsh.model.geo.extrude([(2,s1),(2,s2)],0,0,gaugeThickness,[1,0],recombine=True)
gmsh.model.geo.synchronize()

gmsh.option.setNumber("Mesh.ElementOrder", 2)
gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)

gmsh.model.mesh.generate(3)

delta = 5E-2
top_sur = gmsh.model.getEntitiesInBoundingBox(-gaugeWidth-delta,gaugeHeight+2-delta,-gaugeThickness-delta,
                                              gaugeWidth+delta, gaugeHeight+delta+2,gaugeThickness+delta,
                                              dim=2)

btm_sur = gmsh.model.getEntitiesInBoundingBox(-100-delta,0-delta,0-delta,
                                              100+delta,0+delta,gaugeThickness+delta,
                                              dim=2)

bck_sur = gmsh.model.getEntitiesInBoundingBox(-100,0-delta,0-delta,
                                              100,gaugeHeight+2+delta,0+delta,
                                              dim=2)

vis_sur = gmsh.model.getEntitiesInBoundingBox(-100,0-delta,gaugeThickness-delta,
                                               100,gaugeHeight+delta,gaugeThickness+delta,
                                               dim=2)

vol = gmsh.model.getEntitiesInBoundingBox(-100,-100,-100,
                                        100,100,100,
                                          dim=3)

gmsh.model.geo.addPhysicalGroup(3, [x[1] for x in vol], name='Specimen')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in top_sur], name='Top-BC')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in btm_sur], name='Y-Symm')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in bck_sur], name='Z-Symm')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in vis_sur], name='Visible-Surface')

gmsh.model.geo.synchronize()

exportpath = args.exportpath + '.msh'
gmsh.write(exportpath)
gmsh.finalize()