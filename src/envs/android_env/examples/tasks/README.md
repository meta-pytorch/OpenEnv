# Android Environment Task Definitions

This directory contains task definition files for the Android environment. Tasks define what app to run, how to set it up, and how to reset between episodes.

## Task File Format

Tasks are defined in Protocol Buffer text format (`.textproto`). Here's the basic structure:

```protobuf
id: "task_id"
name: "Task Name"
description: "What this task does"

setup_steps: [
  # Steps to set up the task (run once at environment creation)
  {
    adb_request: {
      install_apk: { filesystem: { path: "/path/to/app.apk" } }
    }
  },
  {
    adb_request: {
      start_activity: { full_activity: "com.example.app/.MainActivity" }
    }
  }
]

reset_steps: [
  # Steps to reset between episodes
  {
    adb_request: {
      force_stop: { package_name: "com.example.app" }
    }
  }
]

expected_app_screen: {
  activity: "com.example.app/.MainActivity"
}

max_episode_sec: 120
max_num_steps: 200
```

## Available Examples

- **calculator_basic.textproto**: Simple calculator app interaction (uses built-in Android calculator)

## Common ADB Requests

### Install APK
```protobuf
adb_request: {
  install_apk: {
    filesystem: { path: "/workspace/apps/myapp.apk" }
  }
}
```

### Start Activity
```protobuf
adb_request: {
  start_activity: {
    full_activity: "com.example.myapp/.MainActivity"
    force_stop: true
  }
}
```

### Force Stop
```protobuf
adb_request: {
  force_stop: {
    package_name: "com.example.myapp"
  }
}
```

### Send Broadcast
```protobuf
adb_request: {
  broadcast: {
    action: "android.intent.action.BOOT_COMPLETED"
  }
}
```

## Creating Custom Tasks

1. **Find your app's package and activity**:
   ```bash
   # Get package name
   adb shell pm list packages | grep myapp

   # Get main activity
   adb shell dumpsys package com.example.myapp | grep -A 1 "android.intent.action.MAIN"
   ```

2. **Create task file**: Copy `calculator_basic.textproto` and modify for your app

3. **Test the task**:
   ```bash
   docker run -it --device /dev/kvm \
     -v $(pwd):/workspace/tasks \
     android-env:latest \
     --task-path /workspace/tasks/my_task.textproto
   ```

4. **Use in training**: Mount your task file when creating the environment

## Task Rewards

Tasks can define custom reward signals based on:
- Screen content matching
- Log events
- Time-based rewards
- Custom reward functions

See the [android_env documentation](https://github.com/deepmind/android_env/blob/main/docs/tasks_guide.md) for full details.

## Tips

- Use `force_stop: true` in `start_activity` to ensure clean state
- Set reasonable `max_episode_sec` to prevent infinite episodes
- Test your task manually with ADB commands first
- Use `wait_for_app_screen` in success conditions to ensure app is ready

## References

- [android_env Tasks Guide](https://github.com/deepmind/android_env/blob/main/docs/tasks_guide.md)
- [android_env Task Proto Definition](https://github.com/deepmind/android_env/blob/main/android_env/proto/task.proto)
- [ADB Commands Reference](https://developer.android.com/tools/adb)
