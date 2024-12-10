const Footer = () => {
	return (
		<div className="border-t border-gray-300 p-4 h-24 mx-4 md:mx-12 lg:mx-24 xl:mx-48 flex flex-col md:flex-row items-center justify-center">
			<input
				type="text"
				placeholder="Type your message..."
				className="border border-gray-300 rounded-l px-4 py-2 w-full mb-2 md:mb-0 md:mr-2"
			/>
			<button className="bg-blue-500 text-white px-4 py-2 rounded-r w-full md:w-auto">
				Submit
			</button>
		</div>
	);
};

export default Footer;
