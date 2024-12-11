import React, { createContext, useContext, useState } from "react";
import PropTypes from "prop-types";
const HeaderContext = createContext();

export const HeaderProvider = ({ children }) => {
	const [headerName, setHeaderName] = useState("CHATTIBOT");

	const updateHeaderName = (newName) => {
		setHeaderName(newName);
	};

	return (
		<HeaderContext.Provider value={{ headerName, updateHeaderName }}>
			{children}
		</HeaderContext.Provider>
	);
};

export const useHeaderContext = () => {
	return useContext(HeaderContext);
};

HeaderProvider.propTypes = {
	children: PropTypes.node,
};
