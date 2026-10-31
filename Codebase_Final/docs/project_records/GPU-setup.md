# 🚀 GPU Setup, Connection & Usage Guide

## 🔌 Access Credentials
- **Username:** saketh
- **Password:** E/sf922)
- **Campus IP:** 10.0.24.7 (port 2222)
- **Public IP:** 61.1.175.170 (port 2222)
- **Expires:** April 5, 2026

---

## 📋 Quick Copy Command - Data Transfer

**Copy "Data Processed" to GPU (FROM CAMPUS):**

```powershell
scp -P 2222 -r "e:\BTP\Data Processed" saketh@10.0.24.7:~/
```

**Copy "Data Processed" to GPU (FROM OUTSIDE CAMPUS):**

```powershell
scp -P 2222 -r "e:\BTP\Data Processed" saketh@61.1.175.170:~/
```

**Result:** Uploads your data to GPU home directory

**Verify on GPU:**
```bash
ls
```

**Expected output:**
```
'Data Processed'   venv
```

---

## 📍 METHOD 1: USING POWERSHELL

### STEP 1: Open PowerShell
```
Windows Key → Type "PowerShell" → Press Enter
```

### STEP 2: Connect to GPU
**FROM CAMPUS:**
```powershell
ssh saketh@10.0.24.7 -p 2222
```

**FROM OUTSIDE CAMPUS:**
```powershell
ssh saketh@61.1.175.170 -p 2222
```

**When prompted:**
```
Password: E/sf922)
```

### STEP 3: Create Environment
```bash
python3 -m venv venv
```

### STEP 4: Activate Environment
```bash
source venv/bin/activate
```

### STEP 5: Install PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### STEP 6: Check GPU is Working
```bash
nvidia-smi
```

### STEP 7: Navigate to Project
```bash
cd Data\ Processed
ls
```

### STEP 8: Run Training
```bash
python3 train.py
```

### STEP 9: Disconnect
```bash
exit
```

---

## 💻 METHOD 2: USING VS CODE

### STEP 1: Open VS Code
```
Click VS Code icon on desktop or Start menu
```

### STEP 2: Open Terminal in VS Code
```
Press Ctrl + Shift + ~
(Or: View → Terminal)
```

### STEP 3: Connect to GPU
**FROM CAMPUS:**
```bash
ssh saketh@10.0.24.7 -p 2222
```

**FROM OUTSIDE CAMPUS:**
```bash
ssh saketh@61.1.175.170 -p 2222
```

**When prompted:**
```
Password: E/sf922)
```

### STEP 4: Create Environment
```bash
python3 -m venv venv
```

### STEP 5: Activate Environment
```bash
source venv/bin/activate
```

### STEP 6: Install PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### STEP 7: Check GPU
```bash
nvidia-smi
```

### STEP 8: Navigate to Project
```bash
cd Data\ Processed
```

### STEP 9: View Files in VS Code
```
Ctrl + K Ctrl + O → Select folder: ~/Data\ Processed
```
*(Now you can edit files in VS Code while connected!)*

### STEP 10: Run Training
```bash
python3 train.py
```

### STEP 11: Monitor GPU (Open New Terminal)
```
Ctrl + Shift + ` (new terminal tab)
```

Then in new terminal:
```bash
watch -n 1 nvidia-smi
```

### STEP 12: Disconnect
```bash
exit
```

---

## 📊 Comparison: PowerShell vs VS Code

| Feature | PowerShell | VS Code |
|---------|-----------|---------|
| Connect | ✅ Easy | ✅ Easy |
| Edit Files | ❌ Text only | ✅ Full editor |
| Monitor GPU | ✅ Yes | ✅ Yes (with split terminals) |
| Upload Files | ✅ Via SCP | ✅ Via SCP or drag-drop |
| Best For | Quick training runs | Development & debugging |

---

## 🎯 QUICK COMMAND REFERENCE

### Connection
- **Connect (Campus):** `ssh saketh@10.0.24.7 -p 2222`
- **Connect (Public):** `ssh saketh@61.1.175.170 -p 2222`

### Environment Setup
- **Create Env:** `python3 -m venv venv`
- **Activate Env:** `source venv/bin/activate`
- **Install PyTorch:** `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### File Operations
- **Check GPU is Available:** `nvidia-smi`
- **List Files:** `ls`
- **Navigate to Data:** `cd Data\ Processed`
- **Delete all files in current folder:** `rm -rf *`
- **Delete specific folder:** `rm -rf folder_name`
- **Delete Data Processed folder:** `rm -rf ~/Data\ Processed`

### Running & Monitoring
- **Run Script:** `python3 train.py`
- **Monitor GPU:** `watch -n 1 nvidia-smi`
- **Disconnect:** `exit`

### Data Transfer (from Windows PowerShell)
- **Upload Data:** `scp -P 2222 -r "e:\BTP\Data Processed" saketh@10.0.24.7:~/`
- **Download Data:** `scp -P 2222 -r saketh@10.0.24.7:~/Data\ Processed "e:\BTP\"`

---

## ⚠️ Important Notes

✅ **DO:**
- Keep password safe
- Request extension before April 5, 2026
- Backup work regularly
- Monitor GPU temperature (should stay < 85°C)
- Check available GPU with `nvidia-smi` before starting training

❌ **DON'T:**
- Change password
- Modify system files
- Install unauthorized software
- Run on unavailable GPUs

---

## 📊 Next Steps After Login

**Navigate to the folder:**
```bash
cd Data\ Processed
```

**Check contents:**
```bash
ls
```
Should show: baby_products/ cosmetics/ electronics/ etc.

**Go back to home:**
```bash
cd ~
```

**Activate your environment:**
```bash
source venv/bin/activate
```

**Run your training:**
```bash
python3 train.py
```

---

## 📈 GPU Usage Snapshots

### Snapshot 1: March 31, 2026 - 03:52:12

```
NVIDIA-SMI 470.256.02   Driver Version: 470.256.02   CUDA Version: 11.4

GPU Status:
┌─────────────────────────────────────────────────┐
│ GPU 0: Tesla K80                                │
│ Temp: 71°C | Power: 145W / 149W                 │
│ Memory: 5444MiB / 11441MiB (47%)                │
│ GPU-Util: 99% | Process: python (5437MiB)      │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ GPU 1: Tesla K80                                │
│ Temp: 51°C | Power: 147W / 149W                 │
│ Memory: 11412MiB / 11441MiB (99%)               │
│ GPU-Util: 99% | Process: python (11403MiB)     │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ GPU 2: Tesla K80                                │
│ Temp: 29°C | Power: 25W / 149W                  │
│ Memory: 7MiB / 11441MiB (<1%)                   │
│ GPU-Util: 0% | Status: IDLE ✅                 │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ GPU 3: Tesla K80                                │
│ Temp: 26°C | Power: 28W / 149W                  │
│ Memory: 7MiB / 11441MiB (<1%)                   │
│ GPU-Util: 0% | Status: IDLE ✅                 │
└─────────────────────────────────────────────────┘

SUMMARY:
✅ GPU 0: TRAINING (PID: 462935)
✅ GPU 1: TRAINING (PID: 3108359)
✅ GPU 2: AVAILABLE
✅ GPU 3: AVAILABLE

Active Processes:
- Python process on GPU 0: 5437MiB
- Python process on GPU 1: 11403MiB
Total In Use: ~16.8GB / 45.7GB (37%)
```

---

**Last Updated:** April 2, 2026
