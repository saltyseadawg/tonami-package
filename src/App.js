import React, { Component } from 'react';
import { Route } from 'react-router-dom';
import './SCSS/App.scss';
import Home from './pages/HomePage';

export default class App extends Component {
  render() {
    return (
      <Home />
    );
  }
}
