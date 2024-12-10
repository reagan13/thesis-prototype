import Header from "./components/Header";
import Footer from "./components/Footer";
const App = () => {
	return (
		<div className="flex flex-col border-red-500 border h-screen">
			<Header />
			<div className="border border-black flex-grow flex gap-10 ">
				<div className="border w-full">h1</div>
				<div className="border w-full">h2</div>
			</div>
			<Footer />
		</div>
	);
};

export default App;
