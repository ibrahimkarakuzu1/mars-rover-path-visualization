import heapq
import math
import numpy as np
import matplotlib.pyplot as plt

# fizik motoru
MASS = 900.0
GRAVITY = 3.721
FRICTION = 0.04

def calculate_energy_cost(p1, p2):
    """"fizik tabanlı enerji maliyeti hesabı"""
    dist_2d = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    delta_h = p2[2] - p1[2]
    dist_3d = math.sqrt(dist_2d**2 + delta_h**2)

    if dist_3d == 0: return 0 
    theta_rad = math.atan2(delta_h, dist_2d)

    gravity_force = MASS * GRAVITY * math.sin(theta_rad)
    friction_force = MASS * GRAVITY * math.cos(theta_rad) * FRICTION
    total_force = gravity_force + friction_force

    if total_force < 0: total_force = 0
    return total_force * dist_3d

#     ARAZİ OLUŞTURUCU  NUMPY 

def generate_terrain(size=50):
    """ Rastgele ama doğal görünümlü bir doğal dağ yaratacağız
     Gaussian fonksiyonu (çan eğrisi ) ile """
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)

    X, Y = np.meshgrid(x, y)

    # gaussian formülü: Ortası yüksek, kenarları alçak 
    # 50 metre yüksekliğinde bir dağ
    Z = 50 * np.exp(-1 * (X**2 + Y**2))
    return Z

#  A* Algoritması

class Node:
    def __init__(self, x,y,z):
        self.x, self.y, self.z = x,y,z
        self.g = float('inf')
        self.f = float('inf')
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, goal):
    return math.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2 + (node.z - goal.z)**2)

def reconstruct_path(current_node):
    path = []
    energy = current_node.g
    while current_node:
        path.append((current_node.x, current_node.y))# sadece xy görselleştirme için
        current_node = current_node.parent
    return path[::-1], energy

def a_star_search(grid, start, goal):
    """grid : NumPy 2D array (yükseklik haritası )"""
    rows, cols = grid.shape
    
    start_node = Node(start[0], start[1], grid[start])
    goal_node = Node(goal[0], goal[1], grid[goal])

    start_node.g = 0
    start_node.f = heuristic(start_node, goal_node)

    open_list = []
    heapq.heappush(open_list, start_node)

    visited = set()
    nodes = {start: start_node}

    print("Rota hesaplanıyor")

    while open_list:
        current = heapq.heappop(open_list)
        
        if(current.x, current.y) == goal:
            return reconstruct_path(current)
        
        visited.add((current.x, current.y))

        #8 yönlü harekt
        neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]

        for dx, dy in neighbors:
            nx, ny = current.x + dx, current.y + dy
            

            if 0 <= nx < rows and 0 <= ny < cols:#harita içinde mi 
                if(nx, ny) in visited: continue

                nz = grid[nx,ny]
                move_cost = calculate_energy_cost(
                    (current.x, current.y, current.z),
                    (nx,ny,nz)
                )

                new_g = current.g + move_cost

                if(nx, ny) not in nodes or new_g < nodes[(nx,ny)].g:
                    neighbor = Node(nx, ny, nz)
                    neighbor.g = new_g
                    neighbor.f = new_g +  heuristic(neighbor, goal_node)
                    neighbor.parent = current
                    nodes[(nx, ny)] = neighbor
                    heapq.heappush(open_list, neighbor)
    return None, 0

# görselleştirme
if __name__ == "__main__":
    #araziyi yaratma
    GRID_SIZE = 50
    terrain = generate_terrain(GRID_SIZE)

    start_pos = (0,0)  #sol alt 
    goal_pos = (GRID_SIZE-1, GRID_SIZE-1) # Sağ Üst- Arada Dağ Var

    # rotayı bul
    path, total_energy = a_star_search(terrain, start_pos ,goal_pos)

    if path:
        print(f"Rota bulund toplam enerji: {total_energy:.2f} J")

        #Matplotlib ile çizdir
        plt.figure(figsize= (10,8))

        #zemin heatmap
        plt.imshow(terrain, cmap= 'terrain', origin = 'lower')
        plt.colorbar(label = "yükseklik(Metre)")

        #Rota çizgi
        #path listesini X ve Y listelerine ayır
        path_y = [p[0] for p in path] # Matris satırları Y eksenidir
        path_x = [p[1] for p in path] # Matris sütunları X eksenidir

        plt.plot(path_x, path_y, color='r', linewidth=2, marker='.', markersize=5, label='Rover Rotası')
        
        # Başlangıç ve Bitiş
        plt.scatter(start_pos[1], start_pos[0], color='lime', s=100, label='Başlangıç', edgecolors='black')
        plt.scatter(goal_pos[1], goal_pos[0], color='red', s=100, marker='*', label='Hedef', edgecolors='black')
        
        plt.title(f"Enerji Tabanlı A* Rotası (Enerji: {total_energy:.0f} J)\nDağın Etrafından Dolaşma Kanıtı")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    else:
        print(" Yol bulunamadı!")
