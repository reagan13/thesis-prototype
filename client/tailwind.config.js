/** @type {import('tailwindcss').Config} */
export default {
	content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
	theme: {
		extend: {
			colors: {
				lightGray: "#f0f1f1", // New light gray color
				lightBlue: "#e4f3ff", // Lightest blue
				blueLight: "#90d5ff", // Light blue
				blueMedium: "#00b2eb", // Medium blue
				blueDark: "#008ab7", // Dark blue
				blue: "#3b82f6", // Blue
				blueDarker: "#006486", // Darker blue
				navy: "#004158", // Navy blue
				darkNavy: "#00202e", // Dark navy
				brightBlue: "#57b9ff", // New bright blue color
			},
			borderColor: {
				DEFAULT: "#3b82f6",
			},
		},
	},
	plugins: [],
};
