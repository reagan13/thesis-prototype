import React, { useState, useRef } from "react";
import { useData } from "../context/DataContext";

const ModelDropdown = () => {
  const { data, setSelectedModel } = useData();
  const models = ["Baseline", "Concat", "Crossattention", "Dense"];
  const [isOpen, setIsOpen] = useState(false);
  const buttonRef = useRef(null);

  const toggleDropdown = () => {
    setIsOpen(!isOpen);
  };

  const handleModelSelect = (model) => {
    setSelectedModel(model);
    setIsOpen(false);
  };

  return (
    <div className="relative inline-block text-left z-20">
      {/* Dropdown Button */}
      <button
        ref={buttonRef}
        onClick={toggleDropdown}
        type="button"
        className="inline-flex justify-center w-full px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        id="options-menu"
        aria-haspopup="true"
        aria-expanded={isOpen}
      >
        {data.selectedModel || "Select Model"}
        <svg
          className="-mr-1 ml-2 h-5 w-5"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      {/* Dropdown Menu with Slide-Down Effect */}
      {isOpen && (
        <div
          style={{ minWidth: `${buttonRef.current?.offsetWidth}px` }} // Adjusted to minWidth
          className="origin-top-right absolute right-0 mt-2 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-20 transform transition-all duration-200 ease-out opacity-100 translate-y-0"
          role="menu"
          aria-orientation="vertical"
          aria-labelledby="options-menu"
        >
          <div className="py-1" role="none">
            {models.map((model, index) => (
              <button
                key={index}
                onClick={() => handleModelSelect(model)}
                className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900 whitespace-nowrap"
                role="menuitem"
              >
                {model}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelDropdown;
