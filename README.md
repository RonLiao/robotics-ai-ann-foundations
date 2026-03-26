# Robotics AI: ANN Foundations 🤖🧠

[![Learning Method: Feynman](https://img.shields.io/badge/Method-Feynman-orange)](https://en.wikipedia.org/wiki/Learning_techniques#Feynman_Technique)
[![Industry: Robotics Startup](https://img.shields.io/badge/Industry-Robotics--Startup-blue)](#)

This repository serves as my technical knowledge base and a living document of my transition from **Android BSP & MCU Engineering** to the field of **Embodied AI**. 

Currently working at a **robotics startup**, I am focused on bridging the gap between high-level AI models (ACT, VLA) and the low-level precision of robotic hardware.

---

## 🛠 My Background
I am a professional **Android BSP & MCU Engineer** specializing in the intersection of software and physical movement:
* **Robotics Industry Experience:** Currently contributing to the development of intelligent robotic systems at a startup.
* **Firmware & Systems:** Expertise in Android BSP optimization, low-level driver development, and MCU firmware.
* **Motion Control:** Focused on DC Motor control and PID tuning to ensure smooth and precise robotic actions.

---

## 📚 Technical Roadmap (Feynman Notes)
This repository contains detailed Markdown notes exported from Notion, covering my deep-dive into AI:

### Phase 1: Neural Network Foundations (Completed ✅)
- [x] **Core Concepts:** Weights, Bias, Activation Functions.
- [x] **Architectures:** FNN, CNN, RNN.
- [x] **Attention Mechanism:** Transformers & Self-Attention.
- [x] **Generative Models:** AE, VAE, CVAE.
- [x] **Embodied AI: ACT**

### Phase 2: Action Policy & Robotics (In-Progress 🚧)
- [ ] **Diffusion Policy**
- [ ] **Vision-Language-Action (VLA) Models**

---

## 🚀 Featured Project: "General Elevator Pressing"
I am currently implementing an end-to-end robotic task using the **SO-101 Robotic Arm** and the **LeRobot** framework.
* **Objective:** Generalize elevator button recognition and pressing across various panel designs.
* **Approach:** Leveraging **ACT** for motion generation while ensuring robust hardware execution through precise **PID Control**.
* **Data Hub:** Real-world demonstrations are hosted on [Hugging Face Datasets](https://huggingface.co/datasets/RonLiao/lerobot-so101-elevator-dataset).
* **Models & Monitoring:** Trained model weights are published on [Hugging Face Models](https://huggingface.co/RonLiao/so101-elevator-act) with training loss metrics available on [WandB](https://wandb.ai/ron-liao-nuwa-robotics/lerobot-so101-elevator).
* **Current Progress:**
  - **[Phase 1: Completed] Practice Task (Circular Magnet Pressing):** Successfully established the end-to-end LeRobot workflow encompassing real-world data collection, multi-process training, and physical robotic arm inference. Surmounted critical infrastructure hurdles including DataLoader limitations, Docker shared memory (`--shm-size`) crashes, and GPU pass-through permissions.
  - **[Phase 2: In Progress] Multi-Task Pressing on Specific Panels:** Initiating the implementation of a Language-Conditioned ACT model architecture. The objective is to utilize natural language commands to drive distinct button-pressing tasks out of a 6-button elevator panel.
  - **[Framework Analysis (Ongoing)]:** Continuously diving into the LeRobot framework architecture, particularly `meta/info.json` properties and Dataset Viewer configurations, to bridge the gap between physical hardware observations and model output dimensions.

---

## 📝 Learning Journey & Blog
For my detailed thought process, including how I apply my BSP experience to solve AI latency and control issues, check out my digital garden:
👉 **[Insert Your Blog Link Here]**

---

## 🤝 Let's Connect
I'm always open to discussing **Embodied AI**, **Android BSP**, or the future of **Robotics**.
- **GitHub:** [https://github.com/RonLiao](https://github.com/RonLiao)
- **Email:** [mib.liao@gmail.com](mailto:mib.liao@gmail.com)

---

## ⚖️ Disclaimer & Attribution
This repository is intended for **non-commercial, personal educational purposes** only. 

* **Content Attribution:** Some technical notes exported from Notion may include diagrams, formulas, or images sourced from academic papers, online courses (e.g., Stanford CS231n), or technical blogs (e.g., iT Help). All copyrights belong to their respective original authors.
* **Fair Use:** The use of such materials is intended to fall under the "Fair Use" doctrine for education and research.
* **Takedown Requests:** If you are the owner of any content used here and would like it to be removed or credited differently, please open an [Issue](https://github.com/RonLiao/robotics-ai-ann-foundations/issues). I will take immediate action to address your concerns.
