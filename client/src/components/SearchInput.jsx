import { useState } from "react"; // Import useState
import TemporaryDrawer from "./TemporaryDrawer";
import { useHeaderContext } from "../context/HeaderContext"; // Import context

const SearchInput = () => {
	const [inputValue, setInputValue] = useState("");
	const { updateHeaderName } = useHeaderContext(); // Use context to get updateHeaderName

	const handleSubmit = () => {
		if (inputValue.trim()) {
			updateHeaderName(inputValue); // Update header name with input value
			setInputValue(""); // Clear the input after submission
		}
	};

	return (
		<div className="border-t border-gray-300 p-4 h-24 mx-4 md:mx-12 lg:mx-24 xl:mx-48 flex flex-col md:flex-row items-center justify-center gap-10">
			<input
				type="text"
				placeholder="Type your message..."
				className="border border-gray-300 rounded-l px-4 py-2 w-full mb-2 md:mb-0 md:mr-2"
				value={inputValue}
				onChange={(e) => setInputValue(e.target.value)} // Update input value
			/>
			<button
				className="bg-blue-500 text-white px-4 py-2 rounded-r w-full md:w-auto"
				onClick={handleSubmit}
			>
				Submit
			</button>
			<TemporaryDrawer />
		</div>
	);
};

export default SearchInput;
