# Insulator Detector - Setup Guide

## Quick Start

### 1. Run the Flask Web App

```powershell
cd "c:\Users\DHARUN\Desktop\insulator project_version2\insular project\insulator"
python app.py
```

Visit: http://localhost:5000

### 2. Run the Flutter Mobile App

```powershell
# Install dependencies
flutter pub get

# Run on connected device/emulator
flutter run
```

## Mobile App Configuration

To connect the mobile app to the Flask server:

1. Open `lib/services/api_service.dart`
2. Update the baseUrl:
   - For Android Emulator: `http://10.0.2.2:5000`
   - For Physical Device: `http://YOUR_COMPUTER_IP:5000`
   - For iOS Simulator: `http://localhost:5000`

## Features

✅ Beautiful gradient UI design
✅ Camera & Gallery support
✅ AI-powered detection
✅ Real-time analysis
✅ Detailed results with visual indicators

---

**Web App**: Accessible at http://127.0.0.1:5000 or http://10.101.141.206:5000
**Mobile App**: Run `flutter run` after `flutter pub get`
