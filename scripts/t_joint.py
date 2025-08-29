import gmsh
import sys

#Geometry variables
radius = 3
width = 10
gaugeThickness = 1.5

h1x = 0
h1y = 0
h1r = 2.5

lc = 2
order = 1
gmsh.initialize()

gmsh.model.add("mod")

p1 = gmsh.model.geo.addPoint(0,0,0,lc) # bottom left
p2 = gmsh.model.geo.addPoint(width/2,0,0,lc) # bottom right
p3 = gmsh.model.geo.addPoint(width/2,width,0,lc) #join to arc
p4 = gmsh.model.geo.addPoint(0,(2*width),0,lc) #top left

p5 = gmsh.model.geo.addPoint(((3/2)*width),(2*width),0,lc) #top right
p6 = gmsh.model.geo.addPoint(((3/2)*width),(width),0,lc) #top right lower

r1s = gmsh.model.geo.addPoint(width/2,width-radius,0,lc)
r1e = gmsh.model.geo.addPoint((width/2)+radius,width,0,lc)
r1c = gmsh.model.geo.addPoint((width/2)+radius,width-radius,0,lc)

l1 = gmsh.model.geo.addLine(p1,p2)
l2 = gmsh.model.geo.addLine(p2,r1s)

l4 = gmsh.model.geo.addLine(p4,p1)
l5 = gmsh.model.geo.addLine(p4,p5)
l6 = gmsh.model.geo.addLine(p5,p6)
l7 = gmsh.model.geo.addLine(p6,r1e)

c1 = gmsh.model.geo.addCircleArc(r1s,r1c,r1e)

cl1 = gmsh.model.geo.addCurveLoop([l1,l2,c1,-l7,-l6,-l5,l4])
s1 = gmsh.model.geo.addPlaneSurface([cl1])

gmsh.model.geo.mesh.setRecombine(2,s1)
gmsh.model.geo.synchronize()

ov = gmsh.model.geo.extrude([(2,s1)],0,0,gaugeThickness,[1],recombine=True)
gmsh.model.geo.synchronize()

gmsh.option.setNumber("Mesh.ElementOrder", 1)

gmsh.model.mesh.generate(3)


delta = 5e-3
left_sur = gmsh.model.getEntitiesInBoundingBox(0-delta,0-delta,-gaugeThickness-delta,
                                              0+delta, 100+delta,gaugeThickness+delta,
                                              dim=2)

btm_sur = gmsh.model.getEntitiesInBoundingBox(-100-delta,0-delta,0-delta,
                                              100+delta,0+delta,gaugeThickness+delta,
                                              dim=2)


bck_sur = gmsh.model.getEntitiesInBoundingBox(-100,-100-delta,0-delta,
                                              100,100+delta,0+delta,
                                              dim=2)

vis_sur = gmsh.model.getEntitiesInBoundingBox(-100,-100-delta,gaugeThickness-delta,
                                               100,100+delta,gaugeThickness+delta,
                                               dim=2)

right_sur = gmsh.model.getEntitiesInBoundingBox(((3/2)*width)-delta,-100-delta,0-delta,
                                               ((3/2)*width)+delta,100+delta,gaugeThickness+delta,
                                               dim=2)

inner_upper = gmsh.model.getEntitiesInBoundingBox((width/2)+radius-delta,width-delta,0-delta,
                                               ((3/2)*width)+delta,width+delta,gaugeThickness+delta,
                                               dim=2)

inner_left = gmsh.model.getEntitiesInBoundingBox((width/2)-delta,0-delta,0-delta,
                                               ((1/2)*width)+delta,width+delta,gaugeThickness+delta,
                                               dim=2)

inner_curve = gmsh.model.getEntitiesInBoundingBox((width/2)-delta,width-radius-delta,0-delta,
                                               (width/2)+radius+delta,width+delta,gaugeThickness+delta,
                                               dim=2)

vol = gmsh.model.getEntitiesInBoundingBox(-100,-100,-100,
                                        100,100,100,
                                          dim=3)

gmsh.model.geo.addPhysicalGroup(3, [x[1] for x in vol], name='Specimen')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in right_sur], name='Right-BC')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in btm_sur], name='Btm-BC')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in bck_sur], name='Z-Symm')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in left_sur], name='X-Symm')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in inner_upper], name='Inner-Upper')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in inner_left], name='Inner-Left')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in inner_curve], name='Inner-Curve')
gmsh.model.geo.addPhysicalGroup(2, [x[1] for x in vis_sur], name='Visible-Surface')

gmsh.model.geo.synchronize()

#if '-nopopup' not in sys.argv:
#    gmsh.fltk.run()
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()
exportpath = 'examples/mesh/tjoint.msh'
gmsh.write(exportpath)
gmsh.finalize()
