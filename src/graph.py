import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap# Use a non-interactive backend for matplotlib
import math

class OptimizedTimeObjectGraph:
    def __init__(self):
        self.time_nodes = {}
        self.object_nodes = {}
        self.time_to_objects = {}
        self.object_to_times = {}
        self.node_counter = 0
        self.color_map = plt.cm.get_cmap('tab10', 10)

    def add_recognition(self, time_value, objects):
        if time_value in self.time_nodes:
            time_id = self.time_nodes[time_value]
        else:
            time_id = self._generate_id()
            self.time_nodes[time_value] = time_id
            self.time_to_objects[time_id] = set()
        
        for obj in objects:
            obj_id = self._get_object_id(obj)
            
            if obj_id not in self.object_nodes:
                self.object_nodes[obj_id] = obj
                self.object_to_times[obj_id] = set()
            
            self.time_to_objects[time_id].add(obj_id)
            self.object_to_times[obj_id].add(time_id)
        
        return time_id

    def merge(self, other_graph):
        for time_value, other_time_id in other_graph.time_nodes.items():
            if time_value in self.time_nodes:
                time_id = self.time_nodes[time_value]
            else:
                time_id = self._generate_id()
                self.time_nodes[time_value] = time_id
                self.time_to_objects[time_id] = set()
        
        for obj_id, obj_info in other_graph.object_nodes.items():
            if obj_id in self.object_nodes:
                existing_obj_id = obj_id
            else:
                existing_obj_id = self._get_object_id(obj_info)
                self.object_nodes[existing_obj_id] = obj_info
                self.object_to_times[existing_obj_id] = set()
        
            for other_time_id in other_graph.object_to_times[obj_id]:
                time_value = self._get_time_value_from_id(other_graph, other_time_id)
                time_id = self.time_nodes[time_value]
                
                self.time_to_objects[time_id].add(existing_obj_id)
                self.object_to_times[existing_obj_id].add(time_id)

    def _get_time_value_from_id(self, graph, time_id):
        for value, id_val in graph.time_nodes.items():
            if id_val == time_id:
                return value
        return None
    
    def _copy(self):
        new_graph = OptimizedTimeObjectGraph()
        new_graph.time_nodes = self.time_nodes.copy()
        new_graph.object_nodes = self.object_nodes.copy()
        new_graph.time_to_objects = {k: v.copy() for k, v in self.time_to_objects.items()}
        new_graph.object_to_times = {k: v.copy() for k, v in self.object_to_times.items()}
        new_graph.node_counter = self.node_counter
        return new_graph

    def _generate_id(self):
        self.node_counter += 1
        return self.node_counter
    
    def _get_object_id(self, obj):
        return str(obj)
    
    def get_time_values(self):
        return list(self.time_nodes.keys())
    
    def get_objects(self):
        return list(self.object_nodes.values())
    
    def visualize(self, title="Optimized Time-Object Graph"):
        G = nx.Graph()

        node_colors = []
        node_labels = {}
        node_sizes = []
        
        for time_value, time_id in self.time_nodes.items():
            G.add_node(f"T{time_id}", type="time")
            node_labels[f"T{time_id}"] = f"T:{time_value}"

            color_idx = time_value % 10
            node_colors.append(self.color_map(color_idx))
            node_sizes.append(1500)
        
        for obj_id, obj_info in self.object_nodes.items():
            G.add_node(f"O{obj_id}", type="object")
            node_labels[f"O{obj_id}"] = obj_info
            node_colors.append("lightgreen")
            node_sizes.append(1200)
        
        for time_value, time_id in self.time_nodes.items():
            for obj_id in self.time_to_objects[time_id]:
                G.add_edge(f"T{time_id}", f"O{obj_id}", color="blue", style="solid")
        
        pos = nx.spring_layout(G, seed=42, k=0.5)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
        
        nx.draw_networkx_labels(G, pos, node_labels, font_size=10, font_weight="bold")
        
        nx.draw_networkx_edges(G, pos, edge_color="blue", width=1.5)
        
        plt.scatter([], [], c='lightgreen', s=1200, label='Object Nodes')
        plt.scatter([], [], c=self.color_map(0), s=1500, label='Time Nodes (Value=0)')
        plt.scatter([], [], c=self.color_map(1), s=1500, label='Time Nodes (Value=1)')
        plt.scatter([], [], c=self.color_map(2), s=1500, label='Time Nodes (Value=2)')
        plt.legend(loc='best', fontsize=8)
        
        plt.title(title, fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def print_state(self):
        print("Time Nodes (value -> id):", self.time_nodes)
        print("Object Nodes:", self.object_nodes)
        print("Time to Objects:")
        for time_value, time_id in self.time_nodes.items():
            objects = [self.object_nodes[o] for o in self.time_to_objects[time_id]]
            print(f"  Time {time_id}({time_value}): {objects}")
        print("Object to Times:")
        for obj_id, times in self.object_to_times.items():
            time_values = [self._get_time_value_from_id(self, t) for t in times]
            print(f"  Object {self.object_nodes[obj_id]}: Times {time_values}")
        print()
    
    def compute_pq_matrices(self, other_graph):
        all_times = set(self.get_time_values()) | set(other_graph.get_time_values())
        all_objects = set(self.get_objects()) | set(other_graph.get_objects())
    
        p11 = p10 = p01 = p00 = 0
        q11 = q10 = q01 = q00 = 0
        
        for time_val in all_times:
            for obj in all_objects:
                in_self = self.has_edge(time_val, obj)
                in_other = other_graph.has_edge(time_val, obj)
                
                if in_self and in_other:
                    q11 += 1
                elif in_self and not in_other:
                    q10 += 1
                elif not in_self and in_other:
                    q01 += 1
                else:
                    q00 += 1
                    # print(time_val, obj, "not in self and not in other")
        
        time_list = sorted(all_times)
        n = len(time_list)
        
        for i in range(n):
            for j in range(i + 1, n):
                t1 = time_list[i]
                t2 = time_list[j]
                
                shared_self = self.shared_objects(t1, t2)
                shared_other = other_graph.shared_objects(t1, t2)

                in_self = shared_self > 0
                in_other = shared_other > 0
                
                if in_self and in_other:
                    p11 += 1
                elif in_self and not in_other:
                    p10 += 1
                    # print(t1, t2, "in self but not in other")
                elif not in_self and in_other:
                    p01 += 1
                else:
                    p00 += 1
        
        total_time_pairs = n * (n - 1) // 2 if n > 1 else 1
        total_obj_pairs = len(all_times) * len(all_objects) if all_objects else 1
        
        p_matrix = [
            [p11 / total_time_pairs, p10 / total_time_pairs],
            [p01 / total_time_pairs, p00 / total_time_pairs]
        ]
        
        q_matrix = [
            [q11 / total_obj_pairs, q10 / total_obj_pairs],
            [q01 / total_obj_pairs, q00 / total_obj_pairs]
        ]
        
        return p_matrix, q_matrix
    
    def has_edge(self, time_value, object_str):
        if time_value not in self.time_nodes:
            return False
        time_id = self.time_nodes[time_value]
        
        obj_id = None
        for oid, obj in self.object_nodes.items():
            if obj == object_str:
                obj_id = oid
                break
        if obj_id is None:
            return False
        
        return obj_id in self.time_to_objects[time_id]
    
    def shared_objects(self, time1, time2):
        objects1 = set()
        objects2 = set()
        
        if time1 in self.time_nodes:
            time_id1 = self.time_nodes[time1]
            objects1 = {self.object_nodes[oid] for oid in self.time_to_objects[time_id1]}
        
        if time2 in self.time_nodes:
            time_id2 = self.time_nodes[time2]
            objects2 = {self.object_nodes[oid] for oid in self.time_to_objects[time_id2]}
        
        return len(objects1 & objects2)

    def calculate_psi(self, p_matrix, q_matrix):
        p11, p10 = p_matrix[0]
        p01, p00 = p_matrix[1]
        
        q11, q10 = q_matrix[0]
        q01, q00 = q_matrix[1]
        
        term1_u = math.sqrt(p11 * p00) if p11 * p00 > 0 else 0
        term2_u = math.sqrt(p10 * p01) if p10 * p01 > 0 else 0
        psi_u = (term1_u - term2_u) ** 2
        
        term1_a = math.sqrt(q11 * q00) if q11 * q00 > 0 else 0
        term2_a = math.sqrt(q10 * q01) if q10 * q01 > 0 else 0
        psi_a = (term1_a - term2_a) ** 2
        
        return psi_u, psi_a 

    def match_score(self, other_graph, tau1, tau2, tau3):
        n = len(set(self.get_time_values()) | set(other_graph.get_time_values()))
        m = len(set(self.get_objects()) | set(other_graph.get_objects()))
        p_matrix, q_matrix = self.compute_pq_matrices(other_graph)
        piu, pia = self.calculate_psi(p_matrix, q_matrix)

        score1 = 1/2 * n * piu + m * pia - math.log(n)
        score2 = n * p_matrix[0][0] + m * pia - math.log(n)
        score3 = -n * math.log(1 - 2 * p_matrix[0][0] + 2 * p_matrix[0][0] ** 2) - m * math.log(1 - 2 * q_matrix[0][0] + 2 * q_matrix[0][0] ** 2) - 2 * math.log(n)
        
        return tau1 * score1 + tau2 * score2 + tau3 * score3

if __name__ == "__main__":
    # graph1 = OptimizedTimeObjectGraph()
    # print("Graph1 - First Recognition:")
    # time_id1 = graph1.add_recognition(1, ["vase", "tv", "window"])
    # graph1.print_state()
    # graph1.visualize("Graph1: Time=1")
    
    # graph2 = OptimizedTimeObjectGraph()
    # print("Graph2 - Second Recognition:")
    # time_id2 = graph2.add_recognition(2, ["vase", "tv", "toilet"])
    # graph2.print_state()
    # graph2.visualize("Graph2: Time=2")
    
    # graph1.merge(graph2)
    # print("Merged Graph:")
    # graph1.print_state()
    # graph1.visualize("Merged Graph")
    
    # print("ger merge time:", graph1.get_time_values())
    # print("get merge object:", graph1.get_objects())
    
    # print("Adding third recognition (time=1):")
    # graph1.add_recognition(3, ["lamp"])
    # graph1.print_state()
    # graph1.visualize("After Third Recognition")

    graph1 = OptimizedTimeObjectGraph()
    graph1.add_recognition(1, ["vase", "tv", "window"])
    graph1.add_recognition(2, ["vase", "tv", "toilet"])
    graph1.add_recognition(3, ["lamp", "window"])
    # graph1.visualize('1')
    
    graph2 = OptimizedTimeObjectGraph()
    graph2.add_recognition(1, ["vase", "tv", "window"])
    graph2.add_recognition(2, ["vase", "tv", "toilet"])
    graph2.add_recognition(3, ["lamp","window"])
    # graph2.visualize('2')
    
    p_matrix, q_matrix = graph1.compute_pq_matrices(graph2)

    piu, pia = graph1.calculate_psi(p_matrix, q_matrix)
    print(f"Psi_u: {piu:.4f}, Psi_a: {pia:.4f}")
    
    x = 1/2 * 3 * piu + 6 * pia - math.log(3)
    xx = 3 * p_matrix[0][0] + 6 * pia - math.log(3)
    
    print(f"x: {x:.4f}")
    print(f"xx: {xx:.4f}")
    

    print("p matrix:")
    print(f"  (1,1): {p_matrix[0][0]:.4f}  (1,0): {p_matrix[0][1]:.4f}")
    print(f"  (0,1): {p_matrix[1][0]:.4f}  (0,0): {p_matrix[1][1]:.4f}")
    
    print("q matrix:")
    print(f"  (1,1): {q_matrix[0][0]:.4f}  (1,0): {q_matrix[0][1]:.4f}")
    print(f"  (0,1): {q_matrix[1][0]:.4f}  (0,0): {q_matrix[1][1]:.4f}")

    graph1.add_recognition(4, ["lamp", "window"])
    graph1.add_recognition(5, ["lamp", "window"])
    graph1.add_recognition(6, ["lamp", "window"])
    graph1.add_recognition(7, ["lamp", "window"])
    graph1.add_recognition(8, ["lamp", "window"])
    graph1.add_recognition(9, ["lamp", "window"])
    

    graph2.add_recognition(4, ["lamp","window"])
    graph2.add_recognition(5, ["lamp","window"])
    graph2.add_recognition(6, ["lamp","window"])
    graph2.add_recognition(7, ["lamp","window"])
    graph2.add_recognition(8, ["lamp","window"])
    graph2.add_recognition(9, ["lamp","window"])
    p_matrix, q_matrix = graph1.compute_pq_matrices(graph2)

    piu, pia = graph1.calculate_psi(p_matrix, q_matrix)
    print(f"Psi_u: {piu:.4f}, Psi_a: {pia:.4f}")

    y = 1/2 * 9 * piu + 6 * pia - math.log(9)
    yy = 9 * p_matrix[0][0] + 6 * pia - math.log(9)
    print(f"y: {y:.4f}")
    print(f"yy: {yy:.4f}")

    print("p matrix:")
    print(f"  (1,1): {p_matrix[0][0]:.4f}  (1,0): {p_matrix[0][1]:.4f}")
    print(f"  (0,1): {p_matrix[1][0]:.4f}  (0,0): {p_matrix[1][1]:.4f}")
    
    print("q matrix:")
    print(f"  (1,1): {q_matrix[0][0]:.4f}  (1,0): {q_matrix[0][1]:.4f}")
    print(f"  (0,1): {q_matrix[1][0]:.4f}  (0,0): {q_matrix[1][1]:.4f}")