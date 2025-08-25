class Timer {
    constructor(callback) {
        this.id = null
        this.time = 0
        this.callback = callback
    }

    start() {
        this.id = setInterval(() => {
            this.time++
            this.callback(this.time)
        }, 1000)
    }

    stop() {
        clearInterval(this.id)
        this.id = null
    }

    reset() {
        this.stop()
        this.time = 0
        this.callback(this.time)
    }
}

export function Time() {
    const [time, setTime] = useState(0);
    const [state] = useState(() => Timer(setTime))
}