import new_client as server
SECRET_KEY='sjLXXU2W66Pc31A1FUe4Fz2mc0k15MTlinvDy1l0Q3GlE5sIEX'

# vec = [0.0, -1.9539671426457027e-12, -2.3110926920993114e-13, 4.585877409544521e-11, -1.7102547340639859e-10, -7.84482094162815e-16, 8.26172948430237e-17, 5.357804104867387e-06, -1.4365553546896628e-07, -6.661912039727877e-10, 9.759048741150336e-13]
best = [0.0, -1.9539671426457027e-12, -2.3110926920993114e-13, 4.585877409544521e-11, -1.7102547340639859e-10, -7.84482094162815e-16, 8.26172948430237e-17, 5.357804104867387e-06, -1.4365553546896628e-07, -6.661912039727877e-10, 9.759048741150336e-13]

second_best = [0.0, 0.0, -2.3072850091119125e-13, 4.612600812254676e-11, -2.1889333684221477e-10, -2.5093706776590157e-16, 1.8564700412166942e-16, -9.180300164214692e-07, -2.2133711264758626e-07, 2.1783703171240047e-09, 0.0]


error = server.get_errors(SECRET_KEY, best)
fitness = abs(error[0]*1 + error[1])

print("Best\n Train error: ", error[0])
print("Validation error: ", error[1])
print("Fitness: ", fitness)
print("\n\n\n")

error = server.get_errors(SECRET_KEY, second_best)
fitness = abs(error[0]*1 + error[1])

print("second_best\n Train error: ", error[0])
print("Validation error: ", error[1])
print("Fitness: ", fitness)
print("\n\n\n")
