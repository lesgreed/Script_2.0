import tkinter as tk
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
import numpy as np
import Surface_rays as geo
import Surface_data as FuD
import NBI_Ports_data_input as Cout
import J_0_test.mconf.mconf as mconf
import os 
import Weight_Fuction.WF_FIDA as WF

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("W7-X cheking")
        self.geometry(f"{1520}x{900}")
        self.data_instance = Data()
        self.Bget = calculus()

        # Global variable for slider value
        self.angle = int(90)
        self.oldangle = int(90)
        self.scale = 10  # New global variable for the second slider
        self.oldscale = 10
        self.delta_s = 0.55

        # Section 1: Sidebar
        self.create_sidebar()

        # Section 3: Additional Widgets
        self.create_additional_widgets()

        # Initialize default values
        self.nbi_optionmenu.set(self.nbi_options[0])
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.update_port_options(self.nbi_options[0])

        self.current_graph = None
        self.all_results = []
        self.data_wf = []
        self.Name_Ports = ['2_1_AEA', '2_1_AEA', '2_1_AEM', '2_1_AEM', '2_1_AET', '2_1_AET'] 
        self.Name_NBI = ['NBI_7', 'NBI_8', 'NBI_7', 'NBI_8', 'NBI_7', 'NBI_8' ]

    def create_sidebar(self):
        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(2, weight=1)
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="W7-X cheking", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.scaling_label = ctk.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))



        self.show_old_button_label= ctk.CTkLabel(self.sidebar_frame, text="Raw data:", anchor="w")    
        self.show_old_button_label.grid(row=4, column=0, padx=10, pady=(10,0))   
        self.show_old_button = ctk.CTkButton(self.sidebar_frame, text="Update data", command=lambda: self.pre_calculate())
        self.show_old_button.grid(row=4, column=0, padx=10, pady=(70,0))
    

    def pre_calculate(self):
        if len(self.all_results) ==0 or self.scale != self.oldscale:
         Result_array = self.data_instance.data_already_input(self.scale)  
         self.all_results = Result_array
         self.data_wf = []
        else:
            self.all_results = [row[:6] for row in self.all_results]

        time = datetime.now().strftime("%H:%M:%S")
        self.textbox.insert("end", f"\n\n [{time}]: Old data ready \n\n ")
        
    def create_additional_widgets(self):
        # create textbox
        self.textbox = ctk.CTkTextbox(self, width=30)  
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # Button to show graph
        self.show_graph_button = ctk.CTkButton(self, text="Add Port", command=lambda: self.dummy_function())
        self.show_graph_button.grid(row=1, column=1, padx=(20, 0), pady=(10, 0), sticky="w")


        # create tabview with two sections
        self.tabview = ctk.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # Section 1: "PORTS and NBI"
        self.tabview.add("PORTS and NBI")
        self.tabview.tab("PORTS and NBI").grid_columnconfigure(0, weight=1)

        # First CTkOptionMenu for NBI selection
        self.nbi_options = self.generate_nbi_options()
        self.nbi_optionmenu_label = ctk.CTkLabel(self.tabview.tab("PORTS and NBI"), text="Select NBI or Gyrotron Launcher", anchor="w")
        self.nbi_optionmenu_label.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="w")
        self.nbi_optionmenu = ctk.CTkOptionMenu(self.tabview.tab("PORTS and NBI"), values=self.nbi_options, command=self.update_port_options)
        self.nbi_optionmenu.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        # Second CTkOptionMenu for Port selection (based on NBI selection)
        self.port_options = []
        self.port_optionmenu_label = ctk.CTkLabel(self.tabview.tab("PORTS and NBI"), text="Select Port", anchor="w")
        self.port_optionmenu_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        self.port_optionmenu = ctk.CTkOptionMenu(self.tabview.tab("PORTS and NBI"), values=self.port_options)
        self.port_optionmenu.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="w")

        # Section 2: "Setting"
        self.tabview.add("Setting")
        self.tabview.tab("Setting").grid_columnconfigure(0, weight=1)

        # Slider in "Setting" section
        self.slider_label = ctk.CTkLabel(self.tabview.tab("Setting"), text="Angle Value:", anchor="w")
        self.slider_label.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="w")

        self.slider = ctk.CTkSlider(self.tabview.tab("Setting"), from_=30, to=90, command=self.slider_event)
        self.slider.set(self.angle)
        self.slider.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="w")

        # Label to display the current slider value
        self.value_label = ctk.CTkLabel(self.tabview.tab("Setting"), text=str(self.angle))
        self.value_label.grid(row=1, column=1, padx=(10, 20), pady=(10, 20), sticky="w")

        # Second slider for values 10 to 100 without intermediate values
        self.second_slider_label = ctk.CTkLabel(self.tabview.tab("Setting"), text="Scale Value:", anchor="w")
        self.second_slider_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")

        self.second_slider = ctk.CTkSlider(self.tabview.tab("Setting"), from_=1, to=10, command=self.second_slider_event)
        self.second_slider.set(self.scale // 10)  # Set initial value based on second_slider_value
        self.second_slider.grid(row=3, column=0, padx=20, pady=(10, 20), sticky="w")

        # Label for Delta s with Greek symbol
        self.delta_s_label = ctk.CTkLabel(self.tabview.tab("Setting"), text="Δs:", anchor="w")
        self.delta_s_label.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="w")

        # Entry for Delta s value
        self.delta_s_entry = ctk.CTkEntry(self.tabview.tab("Setting"), width=100)
        self.delta_s_entry.insert(0, str(self.delta_s))  # Set initial value
        self.delta_s_entry.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="w")

        # Bind the entry to update value automatically when changed
        self.delta_s_entry.bind("<Return>", self.update_delta_s)
        # Label to display the current value of the second slider
        self.second_value_label = ctk.CTkLabel(self.tabview.tab("Setting"), text=str(self.scale))
        self.second_value_label.grid(row=3, column=1, padx=(10, 20), pady=(10, 20), sticky="w")

        # Canvas for displaying the graph
        self.graph_canvas = ctk.CTkCanvas(self, width=800, height=600, bg="white")
        self.graph_canvas.grid(row=0, column=3, padx=(20, 0), pady=(20, 0), sticky="nsew")
    
    def second_slider_event(self, value):
        self.scale = int(value) * 10  
        self.second_value_label.configure(text=str(self.scale))


    def slider_event(self, value):
        self.angle = int(value)
        self.value_label.configure(text=str(self.angle))
    # Method to apply the new Delta s value

    def update_delta_s(self, event):  # Добавлено event=None
        try:
            new_value = float(self.delta_s_entry.get())
            self.delta_s = new_value
            print(f"Delta s updated to: {self.delta_s}")
            # Вставка сообщения в текстовое поле
            self.textbox.insert("end", f"Δs updated to: {self.delta_s}\n\n")
        except ValueError:
            self.textbox.insert("end", "Invalid input. Please enter a numerical value.\n\n")



    def generate_port_options(self, selected_nbi: str):
        nbi_index = int(selected_nbi.split("_")[1])
        if selected_nbi.startswith("CTS"):
         nbi_index = nbi_index+8
        
        nbi_index = nbi_index-1
        valid_indices, extreme_points_1, extreme_points_2, valid_port_names = self.data_instance.port_for_nbi(nbi_index, int(self.angle), self.scale)
        Ports_for_NBI_Index = valid_port_names
        return Ports_for_NBI_Index 

    def update_port_options(self, selected_nbi: str):
     self.port_options = self.generate_port_options(selected_nbi)
     self.port_optionmenu.configure(values=self.port_options)  # Оновлення варіантів
    # Set the initial value for Select Port (if the list is not empty)
     if self.port_options:
            self.port_optionmenu.set(self.port_options[0])
    

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)

    def generate_nbi_options(self):
      return [f"NBI_{i}" if i <= 8 else f"CTS_{i-8}" for i in range(1, 12)]

    def create_result_array_for_port(self, selected_nbi, selected_port):
        data = self.data_instance.data_nbi_ports(selected_nbi, selected_port, self.angle, self.scale)
        for i in range(len(data)):
            self.all_results[i].append(data[i])
        print(self.all_results[0])
        result_for_i = []
        if self.data_wf == []:
         for i in range(len(self.all_results[0])):
            index_nbi = int(self.all_results[0][i].split('_')[-1])-1
            result_for_j = []
            for j in range(len(self.all_results[3][i])):
             if index_nbi<8:
                x_ev = np.linspace(10, 100, 150)
                y_ev = np.linspace(-100, 100, 150)/2.5
                result = WF.weight_Function(self.all_results[4][i][j], self.all_results[3][i][j], x_ev, y_ev)
                result_for_j.append(result)

             if index_nbi>=8:
                x_ev = np.linspace(10, 100, 150)
                y_ev = np.linspace(-100, 100, 150)/2.5
                result = WF.CTS_wf(self.all_results[6][i][j], self.all_results[3][i][j], x_ev, y_ev)  
                result_for_j.append(result)
            result_for_i.append(result_for_j)   
         self.data_wf = result_for_i
        else:
           index_nbi = int(data[0].split('_')[-1])-1
           result_for_j = []
           for j in range(len(data[3])):
             if index_nbi<8:
                x_ev = np.linspace(10, 100, 150)
                y_ev = np.linspace(-100, 100, 150)/2.5
                result = WF.weight_Function(data[4][j], data[3][j], x_ev, y_ev)
                result_for_j.append(result)

             if index_nbi>=8:
                x_ev = np.linspace(10, 100, 150)
                y_ev = np.linspace(-100, 100, 150)/2.5
                result = WF.CTS_wf(data[6][j], data[3][j], x_ev, y_ev)  
                result_for_j.append(result)
           self.data_wf.append(result_for_j)


        


    def generate_and_show_graph(self):
        #User
        selected_nbi = self.nbi_optionmenu.get()
        selected_port = self.port_optionmenu.get()
        
        #Data
        self.Name_NBI.append(selected_nbi)
        self.Name_Ports.append(selected_port)
        self.create_result_array_for_port(selected_nbi, selected_port)
    
        # Clear previous graph
        if self.current_graph:
            self.current_graph.get_tk_widget().destroy()

        # Draw the new graph on the canvas
        self.draw_graph_on_canvas(self.data_wf)

    def dummy_function(self):
        #User 
        selected_nbi = self.nbi_optionmenu.get()
        selected_port = self.port_optionmenu.get()
        
        #Time
        timestamp = datetime.now().strftime("%H:%M:%S")
          
        #Message  
        message = f"[{timestamp}]: Selected Port:    {selected_port}\nSelected:     {selected_nbi}\n\n"
        self.textbox.insert("end", message)


        self.generate_and_show_graph()
        
        
    def draw_graph_on_canvas(self, Result_for_NBI_Port):
        num_arrays = len(Result_for_NBI_Port)
        color = np.array([])
        Matr= np.empty((num_arrays, num_arrays), dtype=object)
            
        # Create a matplotlib figure
        fig, axs = plt.subplots(num_arrays, num_arrays, figsize=(10, 10))

        for i in range(num_arrays):
            for j in range(num_arrays):
                 MATRIX = self.sum(Result_for_NBI_Port[i], Result_for_NBI_Port[j], i, j)
                 min_value = np.min(MATRIX)
                 color = np.append(color, min_value)
                 Matr[i, j]  = MATRIX
        for i in range(num_arrays):
            for j in range(num_arrays):

                One_Matr = Matr[i, j] 
                im = axs[i, j].imshow(One_Matr, cmap='plasma', origin='upper', aspect='auto', vmin=np.min(color), vmax=1.0)

                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        plt.subplots_adjust(wspace=0, hspace=0)
        
        
        # Add colorbar to the last subplot
        cax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [x, y, width, height]
        
        plt.colorbar(im, cax=cax)


        for i in range(num_arrays):
            if num_arrays>=11:
                fonts = 6
            else:
                fonts = 9

            selected_nbi = self.all_results[0][i]
            selected_port = self.Name_Ports[i]
            if selected_nbi[0] == 'N':
               name = 'S'
            else:
               name = 'C'
            axs[num_arrays-1, i].set_xlabel(f'{selected_port[0]}{selected_port[2]}{selected_port[4:]}.{name}{selected_nbi[4]}', fontsize=fonts)
            axs[i, 0].set_ylabel(f'{selected_port[0]}{selected_port[2]}{selected_port[4:]}.{name}{selected_nbi[4]}', fontsize=fonts)



        

        #plt.title('FIDA')
        plt.close('all')

        # Embed the matplotlib figure in the Tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=self.graph_canvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="nsew")

        # Update the current graph reference
        self.current_graph = canvas
        
        
    
            
    def suummmm(self, array_1, array_2, first_nbi_index, second_nbi_index):
     MATRIX = np.zeros((len(array_1), len(array_2)))
     x_ev = np.linspace(10, 100, 150)
     y_ev = np.linspace(-100, 100, 150) / 2.5
     B_max_calc = np.abs(x_ev / y_ev)
     print(len(array_1))
     print(len(array_1[1]))
     print(len(array_1[1][1]))


     for i in range(len(array_1)):
      for j in range(len(array_2)):

        mask = (x_ev < 2.5) & (y_ev < 2.5)  
        product = array_1[i] * array_2[j]
        product *= mask  # Применение маски: зануление элементов, которые не соответствуют условию
        MATRIX[i, j] = np.sum(product)


     max_value = np.max(MATRIX)

     MATRIX = MATRIX / max_value
    
     return MATRIX

    def sum(self, array_1, array_2, first_nbi_index, second_nbi_index):
        MATRIX = np.zeros((len(array_1), len(array_2)))

        # Определите x_ev и y_ev
        x_ev = np.linspace(10, 100, 150)
        y_ev = np.linspace(-100, 100, 150) / 2.5
        x_ev, y_ev = np.meshgrid(x_ev, y_ev)

        # Расчет отношения x_ev / y_ev
        ratio = x_ev / y_ev

        # Значения B_max, s_1 и s_2 для массивов
        B_max_array_1 = self.all_results[8][first_nbi_index]   # Массив или список длиной 10 для array_1
        B_max_array_2 = self.all_results[8][second_nbi_index]  # Массив или список длиной 10 для array_2
        s_1_array = self.all_results[7][first_nbi_index]       # Массив или список s_1 для array_1
        s_2_array = self.all_results[7][second_nbi_index]      # Массив или список s_2 для array_2

        for i in range(len(array_1)):
            for j in range(len(array_2)):

                B_max_i = B_max_array_1[i]
                B_max_j = B_max_array_2[j]
                s_1_i = s_1_array[i]
                s_2_j = s_2_array[j]

                mask_i = (ratio > B_max_i)
                mask_j = (ratio > B_max_j)

                # Проверка условия для обоих слоев: отношение больше B_max
                both_above_B_mask = mask_i & mask_j

                # Проверка условия для обоих слоев: отношение меньше B_max
                both_below_B_mask = np.logical_not(mask_i) & np.logical_not(mask_j)

                # Проверка условия, когда одно меньше, а другое больше
                cross_check_mask = np.logical_not((mask_i & np.logical_not(mask_j)) | (np.logical_not(mask_i) & mask_j))

                # Маска для проверки равенства s_1 и s_2
                equal_s_mask = np.abs(s_1_i - s_2_j) <= self.delta_s

                # Итоговая маска для зануления при обоих отношениях выше B_max и s_1 != s_2
                mask_condition_1 = both_above_B_mask & np.logical_not(equal_s_mask)

                # Итоговая маска для умножения при обоих отношениях выше B_max и s_1 == s_2
                mask_condition_2 = both_above_B_mask & equal_s_mask

                # Итоговая маска для умножения при обоих отношениях ниже B_max
                mask_condition_3 = both_below_B_mask

                # Итоговая маска для зануления, когда соотношения разные для массивов
                mask_condition_4 = np.logical_not(cross_check_mask)

                # Применяем условия к произведению слоев
                product = array_1[i] * array_2[j]
 
                # Обнуление элементов по условиям
                product[mask_condition_1 | mask_condition_4] = 0

                MATRIX[i, j] = np.sum(product)


        # Нормализация матрицы
        max_value = np.max(MATRIX)
        MATRIX /= max_value

        return MATRIX
    






class Data:
        #data_B: [Name NBI; Name Port; Points on NBI; Mag field in this points; Angle between linesight and vec NBI; vec Mag field in points on NBI; angle between vec linesi and magfield]
        #data_B[0]: Name NBI
        #data_B[1]: Name Port
        #data_B[2]: Points on NBI
        #data_B[3]: Mag field in this points
        #data_B[4]: Angle between linesight and vec NBI
        #data_B[5]: vec Mag field in points on NBI
        #data_B[6]: angle between vec linesi and magfield
        #data_B[7]: S
        #data_B[8]: B_max 
        #data_B[9]: results
    def __init__(self):
        self.R_x, self.R_y, self.R_z = FuD.all_point(FuD.read_data()[0])
        self.P_1, self.P_2, self.P_name = Cout.Ports()
        self.NBI_X, self.NBI_Y, self.NBI_Z, self.NBI_uvec_X, self.NBI_uvec_Y, self.NBI_uvec_Z = Cout.NBI()
        self.Bget = calculus()


        # Create 3D surface
        self.surface = geo.create_surface(self.R_x, self.R_y, self.R_z)

        # Get intersections for ports and NBI
        self.new_P_1, *_ = geo.get_intersection_points(self.P_1, self.P_2, self.surface)
        self.new_NBI_start, self.new_NBI_end, *_ = geo.get_intersection_points_NBI(
            self.NBI_X, self.NBI_Y, self.NBI_Z, self.NBI_uvec_X, self.NBI_uvec_Y, self.NBI_uvec_Z, self.surface
        )
        self.valid_indx = []
        for i in range(0, 11):
            valid_indices = geo.pre_NBI_and_PORTS(i, self.new_P_1, self.new_NBI_start, self.new_NBI_end, self.surface)
            # Store the valid port names
            self.valid_indx.append(valid_indices)

    def port_for_nbi(self, NBI_index, angle, scale):
        P_1_for_NBI = self.new_P_1[:, self.valid_indx[NBI_index]]
        P_1_start_for_NBI = self.P_1[:, self.valid_indx[NBI_index]]
        Pname_for_NBI = [self.P_name[i] for i in self.valid_indx[NBI_index]]
        #print(angle)
        valid_indices, extreme_points_1, extreme_points_2, *_ = geo.NBI_and_PORTS(
            P_1_start_for_NBI, NBI_index, P_1_for_NBI, self.new_NBI_start, self.new_NBI_end, self.surface, float(angle))
        valid_port_names = [Pname_for_NBI[i] for i in valid_indices]
        #print(valid_port_names)
        return valid_indices, extreme_points_1, extreme_points_2, valid_port_names

    def data_already_input(self, scale):
        Name_Ports = ['2_1_AEA', '2_1_AEM','2_1_AET'] 
        Name_NBI = ['NBI_7','NBI_8']
        Port_indices = [self.P_name.index(port) for port in Name_Ports if port in self.P_name]
        NBI_indices = [6,7]
        data = [[] for _ in range(6)]
        for i in range(len(NBI_indices)):
            NBI_index_i = NBI_indices[i]
            P_1_for_NBI_i = self.new_P_1[:, Port_indices]
            P_1_start_for_NBI = self.P_1[:, Port_indices]
            valid_indices, extreme_points_1, extreme_points_2, *_ = geo.NBI_and_PORTS(
            P_1_start_for_NBI, NBI_index_i, P_1_for_NBI_i, self.new_NBI_start, self.new_NBI_end, self.surface, float(90))

            for j in range(3):
             data[i*3+j] = (np.array(extreme_points_1[j], dtype=np.float64), np.array(extreme_points_2[j], dtype=np.float64))  # Добавляем к подмассиву


        


        data_B = [[],[],[],[],[],[],[],[],[]]
        for i in range(len(data)):
              points, B_array, B_vec_array, S_array, B_max_array = self.Bget.gets(data[i][0], data[i][1], scale)
              data_B[2].append(points)
              data_B[3].append(B_array)
              data_B[5].append(B_vec_array)
              data_B[7].append(S_array)
              data_B[8].append(B_max_array)

        data_B[0] = ['NBI_7','NBI_7', 'NBI_7','NBI_8', 'NBI_8','NBI_8']
        data_B[1] = ['2_1_AEA', '2_1_AEM','2_1_AET', '2_1_AEA', '2_1_AEM','2_1_AET']
        for i in range(len(data_B[1])):
            index = self.P_name.index(data_B[1][i])
            point_P_2 = [self.new_P_1[0][index],self.new_P_1[1][index],self.new_P_1[2][index]]
            point_P_1 = [self.P_1[0][index],self.P_1[1][index],self.P_1[2][index]]
            angles=[]
            angles_vec_B = []
            for j in range(len(data_B[2][i])):
                angle = geo.check_segment_angle(point_P_1, point_P_2, data_B[2][i][j]*100)
                angles.append(angle)
                
                #angle between vec linesi and magfield
                vector_AB = np.array(point_P_2) - np.array(point_P_1)  
                angle_B = geo.check_angle_2_vec(vector_AB/100, data_B[5][i][j])
                angles_vec_B.append(angle_B)
                 
            data_B[4].append(angles)
            data_B[6].append(angles_vec_B)

        return data_B
    
    def data_nbi_ports(self, nbi, port, angle, scale):
        index = self.P_name.index(port)
        P_1_start = [self.P_1[0][index], self.P_1[1][index], self.P_1[2][index]]
        P_2_end = [self.new_P_1[0][index], self.new_P_1[1][index], self.new_P_1[2][index]]
        
        index_NBI = int(nbi.split('_')[-1])-1
        if nbi.startswith("CTS"):
            index_NBI += 8

        valid_indices, extreme_points_1, extreme_points_2, *_ = geo.NBI_and_PORTS(
            P_1_start, index_NBI, P_2_end, self.new_NBI_start, self.new_NBI_end, self.surface, float(angle))
        print(extreme_points_1)
        points, B_array, B_vec_array, S_array, B_max_array= self.Bget.gets(np.array(extreme_points_1[0], dtype=np.float64), np.array(extreme_points_2[0], dtype=np.float64), scale)

        angles, angles_vec_B=[],[]
        for j in range(len(points)):
                angle = geo.check_segment_angle(P_1_start, P_2_end, points[j]*100)
                angles.append(angle)
                vector_AB = np.array(P_2_end) - np.array(P_1_start)  
                angle_B = geo.check_angle_2_vec(vector_AB/100, B_vec_array[j])
                angles_vec_B.append(angle_B)

        return [nbi, port, points, B_array, angles, B_vec_array, angles_vec_B,S_array, B_max_array]

class calculus():
    def __init__(self):
        pass




    def gets(self, point1, point2, scale):
      points = np.linspace(point1/100, point2/100, scale)
      previous_directory = os.getcwd()
      os.chdir('J_0_test')
      mconf_config = {'B0': 2.525,
                'B0_angle': 0.0,
                'accuracy': 1e-10, #accuracy of magnetic to cartesian coordinat transformation
                'truncation': 1e-10} #trancation of mn harmonics
      eq = mconf.Mconf_equilibrium('w7x-sc1.bc',mconf_config=mconf_config)
      B_array, B_vec_array, S_array, B_max_array= [], [], [], []
      for i in range(len(points)):
         S, vecB = eq.get_B(points[i])
         #if i != 0:
          #print("Distance:", np.sqrt((points[i][0]-points[i-1][0])**2 + (points[i][1]-points[i-1][1])**2 + (points[i][2]-points[i-1][2])**2))
          #print("S", S)
          #print("point1", points[i])
          #print("point2", points[i-1])

         B_max = eq.get_Bmax(S)
         valueB = np.sqrt(vecB[0]**2 + vecB[1]**2 + vecB[2]**2)
         B_array.append(valueB)
         B_vec_array.append(vecB)
         S_array.append(S)
         B_max_array.append(B_max)

      os.chdir(previous_directory)
      return points, B_array, B_vec_array, S_array, B_max_array

        

if __name__ == "__main__":
    app = App()
    app.mainloop()
