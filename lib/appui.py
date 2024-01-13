import threading
import queue
import tkinter as tk
from PIL import Image, ImageTk
import lib.utils as utils

class AppUI(threading.Thread):
    def __init__(self, queue, shared_flag):
        threading.Thread.__init__(self)
        self.shared_flag = shared_flag
        self.queue = queue
        self.trackFeedLabels = {}  # Store labels for updating
        self.daemon = True
        self.trackedPeople = []
        self.start()

    def run(self):
        self.window = tk.Tk()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.title("Tracked People")
        self.window.geometry("500x500")
        self.update_ui()
        self.window.mainloop()

    def on_close(self):
        self.shared_flag[0] = False  # Update the shared flag
        self.window.destroy()

    def update_ui(self):
        try:
            trackedPeople = self.queue.get_nowait()
            self.updateTrackedObjectFeeds(trackedPeople)
        except queue.Empty:
            pass
        finally:
            self.window.after(100, self.update_ui)

    def show_details(self, event, person):
        detail_window = tk.Toplevel(self.window)
        detail_window.title("Details")
        # Display details of the person
        name = str(person.firstName) + " " + str(person.lastName)
        trackerID = person.trackingID
        xOffset = person.xOffset
        yOffset = person.yOffset

        hasFace = person.face is not None
        dbid = "None"

        if hasFace:
            face = person.face

            if face.img is not None:
                faceimg = utils.ImageUtils.ToPil(face.img)
                faceimg = faceimg.resize((180, 180), resample=Image.LANCZOS)
                faceimg = ImageTk.PhotoImage(faceimg)
                dbid = face.identity
                face_label = tk.Label(detail_window, image=faceimg)
                face_label.image = faceimg
                face_label.pack()

        completeString = "Name: " + name + "\n" + "TrackerID: " + str(trackerID) + "\n" + "Has Face: " + str(hasFace) + "\n" + "DBID: " + str(dbid) + "\n" + "X Offset: " + str(xOffset) + "\n" + "Y Offset: " + str(yOffset)

        detail_label = tk.Label(detail_window, text=completeString)  # Placeholder for person details
        detail_label.pack()

    def updateTrackedObjectFeeds(self, trackedPeople):
        self.trackedPeople = trackedPeople

        # Update or create labels for currently tracked people
        for i, person in enumerate(trackedPeople):
            image = person.img
            if image is not None:
                pil_image = utils.ImageUtils.ToPil(image)
                pil_image = pil_image.resize((180, 180), resample=Image.LANCZOS)
                tk_image = ImageTk.PhotoImage(pil_image)

                if i not in self.trackFeedLabels:
                    label = tk.Label(self.window)
                    label.grid(row=i // 5, column=i % 5, padx=5, pady=5)
                    label.bind("<Button-1>", lambda e, person=person: self.show_details(e, person))
                    self.trackFeedLabels[i] = label

                self.trackFeedLabels[i].config(image=tk_image)
                self.trackFeedLabels[i].image = tk_image

        # Hide or clear labels for people no longer tracked
        for i in range(len(trackedPeople), len(self.trackFeedLabels)):
            if i in self.trackFeedLabels:
                self.trackFeedLabels[i].grid_forget()
                self.trackFeedLabels[i].destroy()
                del self.trackFeedLabels[i]
                

# Rest of your code for the queue and thread creation
